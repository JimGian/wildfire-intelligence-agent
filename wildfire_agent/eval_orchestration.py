"""
Orchestration quality eval.

Cases use open-ended or ambiguous queries — the model must decide which tools
to call, in what order, and synthesize a coherent assessment. Checks verify
that substantive claims (not just number echoes) appear in the final report.

NOT a tool-dispatch regression test — that's eval_dispatch.py.

Usage:
    python wildfire_agent/eval_orchestration.py          # 5 runs, default cases
    python wildfire_agent/eval_orchestration.py --runs 3
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_RUNS_DIR = Path(__file__).parent / "eval_runs"
EVAL_RUNS_DIR.mkdir(exist_ok=True)

from wildfire_agent.agent import run_agent


# ─── GPU helpers ─────────────────────────────────────────────────────────────

def _gpu_temp() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=5,
        ).decode().strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def _gpu_vram_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], timeout=5,
        ).decode().strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


# ─── Report-quality helpers ───────────────────────────────────────────────────

def _word_in_report(text: str, *words) -> bool:
    low = text.lower()
    return any(w.lower() in low for w in words)


def _any_num_in_range(text: str, lo: float, hi: float) -> bool:
    """True if any number parsed from text falls in [lo, hi]."""
    for m in re.finditer(r'[\d,]+(?:\.\d+)?', text):
        try:
            n = float(m.group().replace(',', ''))
            if lo <= n <= hi:
                return True
        except ValueError:
            pass
    return False


def _has_value_with_unit(text: str) -> bool:
    """True if a number followed by a physical unit (°C, km/h, %) appears."""
    return bool(re.search(
        r'[\d,]+(?:\.\d+)?\s*(?:°[CF]|km/h|kph|mph|%)',
        text, re.IGNORECASE,
    ))


def _has_ha_value(text: str) -> bool:
    """True if a number followed by ha or hectare(s) appears."""
    return bool(re.search(
        r'[\d,]+(?:\.\d+)?\s*(?:ha\b|hectares?)',
        text, re.IGNORECASE,
    ))


# ─── Agent runner with full trace capture ────────────────────────────────────

def run_traced(query: str, max_turns: int = 10) -> dict:
    tools_called: list[str] = []
    tool_results: dict      = {}
    turns_fired             = [0]

    def _on_call(name, args):
        tools_called.append(name)

    def _on_result(name, result):
        tool_results[name] = result   # last result per tool name

    def _on_turn(idx):
        turns_fired[0] = idx + 1

    error  = None
    report = ""
    t0     = time.time()
    try:
        report = run_agent(
            query, max_turns=max_turns,
            on_tool_call=_on_call,
            on_tool_result=_on_result,
            on_turn=_on_turn,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.time() - t0
    finished_cleanly = (
        error is None
        and bool(report.strip())
        and not report.startswith("Agent did not complete")
    )
    return {
        "report":           report,
        "tools_called":     tools_called,
        "tool_results":     tool_results,
        "llm_turns":        turns_fired[0],
        "tool_count":       len(tools_called),
        "elapsed":          round(elapsed, 1),
        "error":            error,
        "finished_cleanly": finished_cleanly,
    }


# ─── Case definitions ─────────────────────────────────────────────────────────
#
# Queries are open-ended; no tool names appear in the query text.
# SHOULD lists reflect the ideal orchestration path. Quality checks verify
# synthesised claims — not just that a number from the tool output was echoed.
#
# Smoke-test reference (qwen2.5:7b, temp=0.2):
#   Query: "How bad was the Evros wildfire in August 2023?"
#   Tools: lookup_historical_context → fetch_satellite_imagery →
#          run_burn_scar_model → query_firms_active_fires → query_weather_conditions
#   Turns: 6, report included burned_ha, VIIRS count, temp, severity judgment.

CASES = [
    # ── 1. Open severity query ────────────────────────────────────────────────
    # Deliberately reuses the smoke-test query for direct comparison.
    {
        "id": "open_severity_query",
        "query": "How bad was the Evros wildfire in August 2023?",
        "must": [
            ("finished_cleanly",  lambda t: t["finished_cleanly"]),
            ("non_empty_report",  lambda t: len(t["report"].strip()) > 100),
            ("under_turn_limit",  lambda t: t["llm_turns"] < 12),
        ],
        "should": [
            "lookup_historical_context",
            "fetch_satellite_imagery",
            "run_burn_scar_model",
            "query_firms_active_fires",
            "query_weather_conditions",
        ],
        "quality": [
            # Names the fire / region — "evros" excluded (it's in the query text)
            ("fire_named",
             lambda t: _word_in_report(t["report"], "dadia", "national park", "northern greece")),
            # Mentions area on the order of thousands of ha — checks synthesis,
            # not a specific number echo (any value 1 000–100 000 ha qualifies)
            ("area_in_range",
             lambda t: _has_ha_value(t["report"]) and _any_num_in_range(t["report"], 1_000, 100_000)),
            # Report mentions weather context — requires weather tool OR RAG
            ("weather_context",
             lambda t: _word_in_report(t["report"],
                 "temperature", "wind", "humidity", "heat", "drought", "hot")),
            # Report mentions hotspot or thermal observations — specific enough
            # that it won't appear without FIRMS data or a RAG doc mentioning it
            ("hotspot_or_thermal",
             lambda t: _word_in_report(t["report"],
                 "hotspot", "firms", "viirs", "thermal", "detection")),
            # Report makes a severity judgement — not a number, a qualitative claim
            ("severity_judgment",
             lambda t: _word_in_report(t["report"],
                 "severe", "extreme", "catastrophic", "largest", "devastating",
                 "unprecedented", "worst")),
        ],
    },

    # ── 2. Comparative historical query ──────────────────────────────────────
    {
        "id": "comparative_historical_query",
        "query": (
            "What was the worst Greek wildfire of 2023 and how does it compare to "
            "historical Greek wildfires? Was the weather unusually severe?"
        ),
        "must": [
            ("finished_cleanly",  lambda t: t["finished_cleanly"]),
            ("non_empty_report",  lambda t: len(t["report"].strip()) > 100),
            ("under_turn_limit",  lambda t: t["llm_turns"] < 12),
        ],
        # lookup_historical_context ideally called ≥2×: once to id the 2023 worst,
        # once for comparative context. SHOULD is binary per tool (called or not).
        "should": [
            "lookup_historical_context",
            "query_weather_conditions",
            "fetch_satellite_imagery",
            "run_burn_scar_model",
        ],
        "quality": [
            # Names a specific 2023 Greek fire
            ("greek_2023_fire_named",
             lambda t: _word_in_report(t["report"], "evros", "dadia", "rhodes")),
            # References at least one historical Greek fire by name or distinctive year
            # (2018=Mati, 2007=Peloponnese, 2021=Evia)
            ("historical_fire_referenced",
             lambda t: _word_in_report(t["report"],
                 "mati", "peloponnese", "evia", "2018", "2007", "2021")),
            # Makes an explicit comparison claim — requires synthesis, not just listing
            ("comparison_made",
             lambda t: _word_in_report(t["report"],
                 "larger", "smaller", "worse", "compared", "exceeded",
                 "unprecedented", "surpassed", "more severe", "less severe", "biggest")),
            # Report includes at least one weather value with a physical unit
            # (temperature °C, wind km/h, or humidity %) — not just a word
            ("weather_specific_value",
             lambda t: _has_value_with_unit(t["report"])),
        ],
    },

    # ── 3. Ambiguous fire query ───────────────────────────────────────────────
    # Vague query: model must identify the fire and commit without being told.
    {
        "id": "ambiguous_fire_query",
        "query": "Tell me about the 2023 fire in northern Greece.",
        "must": [
            ("finished_cleanly",  lambda t: t["finished_cleanly"]),
            ("non_empty_report",  lambda t: len(t["report"].strip()) > 100),
            ("under_turn_limit",  lambda t: t["llm_turns"] < 12),
        ],
        "should": [
            "lookup_historical_context",    # should identify fire before fetching imagery
        ],
        "quality": [
            # Commits to a specific fire (does not stay vague)
            ("fire_identified",
             lambda t: _word_in_report(t["report"], "evros", "dadia")),
            # Does not refuse or ask for clarification
            ("no_refusal",
             lambda t: not _word_in_report(t["report"],
                 "i don't know", "please specify", "could you clarify",
                 "please clarify", "need more information", "which fire")),
            # Includes at least one quantitative finding: area in ha
            ("quantitative_finding",
             lambda t: _has_ha_value(t["report"])),
            # Mentions a month or date (not just the year from the query)
            ("date_or_month",
             lambda t: _word_in_report(t["report"],
                 "august", "july", "september", "aug", "jul", "sep")
             or bool(re.search(r'\b\d{4}-\d{2}', t["report"]))),
        ],
    },

    # ── 4. Weather anomaly query ──────────────────────────────────────────────
    {
        "id": "weather_anomaly_query",
        "query": (
            "Were the weather conditions during the 2023 Evros fire unusually dangerous "
            "compared to normal August conditions?"
        ),
        "must": [
            ("finished_cleanly",  lambda t: t["finished_cleanly"]),
            ("non_empty_report",  lambda t: len(t["report"].strip()) > 100),
            ("under_turn_limit",  lambda t: t["llm_turns"] < 12),
        ],
        "should": [
            "query_weather_conditions",
            "lookup_historical_context",
        ],
        "quality": [
            # Report includes a specific temperature in a plausible °C range for Greece
            ("temp_value_present",
             lambda t: _any_num_in_range(t["report"], 20, 50)),
            # Report includes a value with a physical unit (°C, km/h, or %)
            # — verifies the value came from a tool, not just a vague claim
            ("value_with_unit",
             lambda t: _has_value_with_unit(t["report"])),
            # Makes a qualitative danger or risk judgement
            ("danger_judgment",
             lambda t: _word_in_report(t["report"],
                 "dangerous", "extreme", "severe", "unusual", "high risk",
                 "elevated", "hazardous", "critical")),
            # Either acknowledges a limitation (no historical baseline available)
            # or provides a comparison from RAG context
            ("limitation_or_baseline",
             lambda t: _word_in_report(t["report"],
                 "baseline", "normal", "average", "typical", "usually",
                 "no data", "limited", "without historical")),
        ],
    },
]


# ─── Eval runner ─────────────────────────────────────────────────────────────

TEMP_PAUSE_C   = 80
VRAM_PAUSE_MIB = 12_000   # slightly more relaxed than dispatch eval


def _check_gpu(run_label: str) -> bool:
    """Prints warning and pauses if thermal/VRAM limits exceeded."""
    temp = _gpu_temp()
    vram = _gpu_vram_mib()
    flagged = False
    if temp is not None and temp > TEMP_PAUSE_C:
        print(f"\n*** THERMAL WARNING: {temp}°C > {TEMP_PAUSE_C}°C after {run_label} — pausing 60s ***")
        time.sleep(60)
        flagged = True
    if vram is not None and vram > VRAM_PAUSE_MIB:
        print(f"\n*** VRAM WARNING: {vram} MiB > {VRAM_PAUSE_MIB} MiB after {run_label} ***")
        flagged = True
    return not flagged


def run_eval_orchestration(n_runs: int = 5) -> None:
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_traces: dict = {}
    case_summaries   = []

    total_must_pass = total_must_total = 0
    total_should_pass = total_should_total = 0
    total_quality_pts = total_quality_max = 0

    print(f"Orchestration quality eval — {len(CASES)} cases × {n_runs} runs  [{timestamp}]")
    print("=" * 65)

    for case in CASES:
        cid     = case["id"]
        traces  = []
        all_traces[cid] = traces

        must_checks    = case["must"]
        should_tools   = case["should"]
        quality_checks = case["quality"]

        run_must_passes   = []
        run_should_hits   = []
        run_quality_pts   = []
        run_turns         = []
        run_tool_counts   = []
        run_elapsed       = []
        first_partial     = None

        print(f"\nCASE: {cid}")

        for run_i in range(n_runs):
            label = f"{cid} run {run_i+1}"
            t = run_traced(case["query"])
            traces.append({
                "run": run_i + 1,
                "query": case["query"],
                **t,
            })

            must_results = [fn(t) for _, fn in must_checks]
            run_must_passes.append(all(must_results))

            should_hit = {tool: (tool in t["tools_called"]) for tool in should_tools}
            run_should_hits.append(should_hit)

            q_pts = sum(1 for _, fn in quality_checks if fn(t))
            run_quality_pts.append(q_pts)

            run_turns.append(t["llm_turns"])
            run_tool_counts.append(t["tool_count"])
            run_elapsed.append(t["elapsed"])

            if not all(must_results) and first_partial is None:
                first_partial = {
                    "run": run_i + 1, "kind": "MUST_fail",
                    "must_results": dict(zip([n for n,_ in must_checks], must_results)),
                    "tools_called": t["tools_called"],
                    "error": t["error"],
                    "report_snippet": t["report"][:500],
                }
            elif q_pts < len(quality_checks) and first_partial is None:
                missed_q = [n for (n, fn) in quality_checks if not fn(t)]
                first_partial = {
                    "run": run_i + 1, "kind": "quality_partial",
                    "quality_pts": f"{q_pts}/{len(quality_checks)}",
                    "missed_quality": missed_q,
                    "tools_called": t["tools_called"],
                    "report_snippet": t["report"][:500],
                }

            must_ok = "M✓" if all(must_results) else "M✗"
            s_hits  = sum(should_hit.values())
            q_str   = f"Q{q_pts}/{len(quality_checks)}"
            print(f"  run {run_i+1}/{n_runs}  {must_ok}  S{s_hits}/{len(should_tools)}  {q_str}"
                  f"  {t['tool_count']}tools  {t['llm_turns']}turns  {t['elapsed']}s"
                  + (f"  [ERR: {t['error'][:60]}]" if t['error'] else ""))

            _check_gpu(label)

        must_passes  = sum(run_must_passes)
        should_agg   = {tool: sum(rh[tool] for rh in run_should_hits) for tool in should_tools}
        mean_quality = sum(run_quality_pts) / n_runs
        mean_turns   = sum(run_turns) / n_runs
        max_turns_   = max(run_turns)
        mean_tools   = sum(run_tool_counts) / n_runs
        mean_elapsed = sum(run_elapsed) / n_runs

        total_must_pass  += must_passes;  total_must_total  += n_runs
        for tool in should_tools:
            total_should_pass += should_agg[tool]; total_should_total += n_runs
        total_quality_pts += sum(run_quality_pts)
        total_quality_max += n_runs * len(quality_checks)

        print(f"  MUST:           {must_passes}/{n_runs} passed")
        print(f"  SHOULD:         " +
              "  ".join(f"{tool[:12]} ({should_agg[tool]}/{n_runs})"
                        for tool in should_tools))
        print(f"  REPORT QUALITY: mean {mean_quality:.1f}/{len(quality_checks)}")
        print(f"  ITERATIONS:     mean {mean_turns:.1f}, max {max_turns_}")
        print(f"  TOOL CALLS:     mean {mean_tools:.1f}")
        print(f"  RUNTIME:        mean {mean_elapsed:.0f}s")
        if first_partial:
            if first_partial["kind"] == "MUST_fail":
                print(f"  FIRST PARTIAL: run {first_partial['run']} MUST FAILED "
                      f"{first_partial['must_results']}")
                if first_partial["error"]:
                    print(f"    error: {first_partial['error'][:120]}")
            else:
                missed = first_partial["missed_quality"]
                print(f"  FIRST PARTIAL: run {first_partial['run']} "
                      f"{first_partial['quality_pts']} quality, missed: {missed}")
                print(f"    tools: {first_partial['tools_called']}")
                print(f"    report: {first_partial['report_snippet'][:300]}")

        case_summaries.append({
            "id": cid, "must": f"{must_passes}/{n_runs}",
            "should": should_agg, "mean_quality": round(mean_quality, 2),
            "quality_max": len(quality_checks), "mean_turns": round(mean_turns, 2),
            "max_turns": max_turns_, "mean_tools": round(mean_tools, 2),
            "mean_elapsed_s": round(mean_elapsed, 1), "first_partial": first_partial,
        })

    out_path = EVAL_RUNS_DIR / f"eval_orchestration_{timestamp}.json"
    out_path.write_text(
        json.dumps({"timestamp": timestamp, "n_runs": n_runs,
                    "cases": case_summaries, "traces": all_traces},
                   default=str, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("OVERALL SUMMARY  [orchestration quality]")
    print(f"  MUST pass rate:    {total_must_pass}/{total_must_total} "
          f"({total_must_pass/total_must_total:.0%})")
    print(f"  SHOULD pass rate:  {total_should_pass}/{total_should_total} "
          f"({total_should_pass/total_should_total:.0%})")
    print(f"  Mean report quality: {total_quality_pts/total_quality_max:.0%} "
          f"({total_quality_pts}/{total_quality_max})")
    print(f"\nFull traces saved to: {out_path}")

    must_failures = [s for s in case_summaries if s["must"] != f"{n_runs}/{n_runs}"]
    if must_failures:
        print("\n*** MUST FAILURES ***")
        for s in must_failures:
            print(f"  {s['id']}: {s['must']}")
            if s["first_partial"]:
                print(f"    {json.dumps(s['first_partial'], default=str)[:300]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of agent runs per case (default 5)")
    args = parser.parse_args()
    run_eval_orchestration(args.runs)
