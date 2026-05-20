"""
Regression test for tool dispatch.

Each case uses a prescriptive query that names the tools to call, and verifies
the model calls them and surfaces their results in the report.

NOT a measure of orchestration quality — that's eval_orchestration.py.

Usage:
    python wildfire_agent/eval_dispatch.py          # 5 runs, default cases
    python wildfire_agent/eval_dispatch.py --runs 3  # custom run count
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

def _num_in_report(text: str, value: float, tol: float = 0.10) -> bool:
    """True if any number in text is within tol of value (relative tolerance)."""
    lo, hi = value * (1 - tol), value * (1 + tol)
    for m in re.finditer(r'[\d,]+(?:\.\d+)?', text):
        try:
            n = float(m.group().replace(',', ''))
            if lo <= n <= hi:
                return True
        except ValueError:
            pass
    return False


def _word_in_report(text: str, *words) -> bool:
    low = text.lower()
    return any(w.lower() in low for w in words)


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
        turns_fired[0] = idx + 1     # 1-indexed count of LLM calls so far

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
# Known deterministic tool outputs (from prior confirmed runs):
#   evros  run_burn_scar_model: burned_ha.moderate_0.27=8675.1, fire_prob=0.8238
#   rhodes run_burn_scar_model: burned_ha.moderate_0.27=11236.5, fire_prob=0.9713
#   evros  query_weather_conditions("2023-08-01"): temp_max=31.0, wind=11.6, humidity=61.0
#
# FIRMS counts are API-backed and vary slightly; check presence/range rather than exact value.

CASES = [
    # ── 1. Evros burn assessment ──────────────────────────────────────────────
    {
        "id": "evros_burn_assessment",
        "query": (
            "Assess the burned area and fire severity for the 2023 Evros wildfire. "
            "Fetch the satellite imagery first, then run the burn scar model."
        ),
        "must": [
            ("finished_cleanly",    lambda t: t["finished_cleanly"]),
            ("non_empty_report",    lambda t: len(t["report"].strip()) > 50),
            ("under_turn_limit",    lambda t: t["llm_turns"] < 12),
        ],
        "should": [
            "fetch_satellite_imagery",
            "run_burn_scar_model",
        ],
        "quality": [
            # burned_ha.moderate_0.27 = 8675.1
            ("burned_ha_in_report",   lambda t: _num_in_report(t["report"], 8675.1)),
            # fire_probability = 0.8238
            ("fire_prob_in_report",   lambda t: _num_in_report(t["report"], 0.8238, tol=0.15)),
            # region named
            ("evros_named",           lambda t: _word_in_report(t["report"], "evros")),
            # unit mentioned
            ("ha_unit_mentioned",     lambda t: _word_in_report(t["report"], "ha", "hectare")),
        ],
    },

    # ── 2. Rhodes burn assessment ─────────────────────────────────────────────
    {
        "id": "rhodes_burn_assessment",
        "query": (
            "Assess the burned area and fire severity for the 2023 Rhodes wildfire. "
            "Fetch the satellite imagery first, then run the burn scar model."
        ),
        "must": [
            ("finished_cleanly",    lambda t: t["finished_cleanly"]),
            ("non_empty_report",    lambda t: len(t["report"].strip()) > 50),
            ("under_turn_limit",    lambda t: t["llm_turns"] < 12),
        ],
        "should": [
            "fetch_satellite_imagery",
            "run_burn_scar_model",
        ],
        "quality": [
            # burned_ha.moderate_0.27 = 11236.5
            ("burned_ha_in_report",   lambda t: _num_in_report(t["report"], 11236.5)),
            # fire_probability = 0.9713
            ("fire_prob_in_report",   lambda t: _num_in_report(t["report"], 0.9713, tol=0.15)),
            ("rhodes_named",          lambda t: _word_in_report(t["report"], "rhodes")),
            ("ha_unit_mentioned",     lambda t: _word_in_report(t["report"], "ha", "hectare")),
        ],
    },

    # ── 3. Imagery availability check ────────────────────────────────────────
    {
        "id": "evros_imagery_check",
        "query": (
            "Confirm that Sentinel-2 satellite imagery is available for the evros region "
            "and report which time periods (pre-fire, during, post-fire) are cached on disk."
        ),
        "must": [
            ("finished_cleanly",    lambda t: t["finished_cleanly"]),
            ("non_empty_report",    lambda t: len(t["report"].strip()) > 30),
            ("under_turn_limit",    lambda t: t["llm_turns"] < 12),
        ],
        "should": [
            "fetch_satellite_imagery",
        ],
        "quality": [
            ("pre_fire_mentioned",  lambda t: _word_in_report(t["report"], "pre_fire", "pre-fire", "pre fire")),
            ("post_fire_mentioned", lambda t: _word_in_report(t["report"], "post_fire", "post-fire", "post fire")),
            ("evros_named",         lambda t: _word_in_report(t["report"], "evros")),
            ("cached_or_available", lambda t: _word_in_report(t["report"], "cached", "available", "exist", "found", "disk")),
        ],
    },

    # ── 4. FIRMS hotspot count ────────────────────────────────────────────────
    {
        "id": "evros_firms_hotspots",
        "query": (
            "How many active fire hotspots were detected in the Evros region during "
            "August 2023? Use FIRMS thermal observations to cross-validate fire activity."
        ),
        "must": [
            ("finished_cleanly",    lambda t: t["finished_cleanly"]),
            ("non_empty_report",    lambda t: len(t["report"].strip()) > 30),
            ("under_turn_limit",    lambda t: t["llm_turns"] < 12),
        ],
        "should": [
            "query_firms_active_fires",
            "lookup_historical_context",    # should look up ignition date first
        ],
        "quality": [
            # Any positive integer detection count mentioned
            ("detection_count_in_report", lambda t: bool(re.search(r'\b[1-9]\d{1,4}\b', t["report"]))),
            ("hotspot_keyword",           lambda t: _word_in_report(t["report"], "hotspot", "detection", "VIIRS", "FIRMS")),
            ("evros_named",               lambda t: _word_in_report(t["report"], "evros")),
            ("august_2023_mentioned",     lambda t: "2023" in t["report"] and _word_in_report(t["report"], "august", "aug", "08-")),
        ],
    },

    # ── 5. Weather conditions ─────────────────────────────────────────────────
    {
        "id": "evros_weather_aug1",
        "query": "What were the meteorological conditions in the Evros region on August 1st, 2023?",
        "must": [
            ("finished_cleanly",    lambda t: t["finished_cleanly"]),
            ("non_empty_report",    lambda t: len(t["report"].strip()) > 30),
            ("under_turn_limit",    lambda t: t["llm_turns"] < 12),
        ],
        "should": [
            "query_weather_conditions",
        ],
        "quality": [
            # temp_max = 31.0°C
            ("temp_in_report",      lambda t: _num_in_report(t["report"], 31.0, tol=0.15)),
            # wind_speed = 11.6 km/h
            ("wind_in_report",      lambda t: _num_in_report(t["report"], 11.6, tol=0.15)),
            ("evros_named",         lambda t: _word_in_report(t["report"], "evros")),
            ("date_mentioned",      lambda t: "august" in t["report"].lower() or "aug" in t["report"].lower() or "2023-08-01" in t["report"]),
        ],
    },

    # ── 6. Historical context retrieval ──────────────────────────────────────
    {
        "id": "historical_northern_greece",
        "query": "Find historical context about large fires in northern Greece forests.",
        "must": [
            ("finished_cleanly",    lambda t: t["finished_cleanly"]),
            ("non_empty_report",    lambda t: len(t["report"].strip()) > 50),
            ("under_turn_limit",    lambda t: t["llm_turns"] < 12),
        ],
        "should": [
            "lookup_historical_context",
        ],
        "quality": [
            # Should mention at least one specific fire name or location
            ("fire_or_location_named", lambda t: _word_in_report(t["report"],
                "evros", "dadia", "evia", "peloponnese", "mati", "northern greece")),
            # Should mention at least one year
            ("year_mentioned",         lambda t: any(y in t["report"] for y in ["2021", "2023", "2007", "2018"])),
            # Should mention scale — hectares or area
            ("area_mentioned",         lambda t: _word_in_report(t["report"], "ha", "hectare", "km²", "area", "burned")),
            # Should reference a source or doc title
            ("title_or_source",        lambda t: _word_in_report(t["report"],
                "effis", "copernicus", "wwf", "civil protection", "wildfire", "fire season")),
        ],
    },
]


# ─── Eval runner ─────────────────────────────────────────────────────────────

TEMP_PAUSE_C  = 80
VRAM_PAUSE_MIB = 11_500   # conservative ceiling given display overhead


def _check_gpu(run_label: str) -> bool:
    """Returns False and prints warning if thermal/VRAM limits exceeded."""
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


def run_eval_dispatch(n_runs: int = 5) -> None:
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_traces: dict = {}
    case_summaries   = []

    total_must_pass = total_must_total = 0
    total_should_pass = total_should_total = 0
    total_quality_pts = total_quality_max = 0

    print(f"Tool-dispatch regression eval — {len(CASES)} cases × {n_runs} runs  [{timestamp}]")
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
                first_partial = {"run": run_i + 1, "kind": "MUST_fail",
                                 "must_results": dict(zip([n for n,_ in must_checks], must_results)),
                                 "tools_called": t["tools_called"],
                                 "error": t["error"],
                                 "report_snippet": t["report"][:400]}
            elif q_pts < len(quality_checks) and first_partial is None:
                missed_q = [n for (n,fn) in quality_checks if not fn(t)]
                first_partial = {"run": run_i + 1, "kind": "quality_partial",
                                 "quality_pts": f"{q_pts}/{len(quality_checks)}",
                                 "missed_quality": missed_q,
                                 "tools_called": t["tools_called"],
                                 "report_snippet": t["report"][:400]}

            must_ok  = "M✓" if all(must_results) else "M✗"
            s_hits   = sum(should_hit.values())
            q_str    = f"Q{q_pts}/{len(quality_checks)}"
            print(f"  run {run_i+1}/{n_runs}  {must_ok}  S{s_hits}/{len(should_tools)}  {q_str}"
                  f"  {t['tool_count']}tools  {t['llm_turns']}turns  {t['elapsed']}s"
                  + (f"  [ERR: {t['error'][:60]}]" if t['error'] else ""))

            _check_gpu(label)

        must_passes    = sum(run_must_passes)
        should_agg     = {tool: sum(rh[tool] for rh in run_should_hits) for tool in should_tools}
        mean_quality   = sum(run_quality_pts) / n_runs
        mean_turns     = sum(run_turns) / n_runs
        max_turns      = max(run_turns)
        mean_tools     = sum(run_tool_counts) / n_runs
        mean_elapsed   = sum(run_elapsed) / n_runs

        total_must_pass  += must_passes;  total_must_total  += n_runs
        for tool in should_tools:
            total_should_pass += should_agg[tool]; total_should_total += n_runs
        total_quality_pts += sum(run_quality_pts)
        total_quality_max += n_runs * len(quality_checks)

        print(f"  MUST:          {must_passes}/{n_runs} passed")
        print(f"  SHOULD:        " +
              "  ".join(f"{tool.split('_')[1][:6]} ({should_agg[tool]}/{n_runs})"
                        for tool in should_tools))
        print(f"  REPORT QUALITY: mean {mean_quality:.1f}/{len(quality_checks)} values surfaced")
        print(f"  ITERATIONS:    mean {mean_turns:.1f}, max {max_turns}")
        print(f"  TOOL CALLS:    mean {mean_tools:.1f}")
        print(f"  RUNTIME:       mean {mean_elapsed:.0f}s")
        if first_partial:
            if first_partial["kind"] == "MUST_fail":
                print(f"  FIRST PARTIAL: run {first_partial['run']} MUST FAILED "
                      f"{first_partial['must_results']}")
                if first_partial["error"]:
                    print(f"    error: {first_partial['error'][:120]}")
            else:
                missed = first_partial["missed_quality"]
                print(f"  FIRST PARTIAL: run {first_partial['run']} {first_partial['quality_pts']} "
                      f"quality, missed: {missed}")
                print(f"    tools: {first_partial['tools_called']}")
                print(f"    report: {first_partial['report_snippet'][:200]}")

        case_summaries.append({
            "id": cid, "must": f"{must_passes}/{n_runs}",
            "should": should_agg, "mean_quality": round(mean_quality, 2),
            "quality_max": len(quality_checks), "mean_turns": round(mean_turns, 2),
            "max_turns": max_turns, "mean_tools": round(mean_tools, 2),
            "mean_elapsed_s": round(mean_elapsed, 1), "first_partial": first_partial,
        })

    out_path = EVAL_RUNS_DIR / f"eval_dispatch_{timestamp}.json"
    out_path.write_text(
        json.dumps({"timestamp": timestamp, "n_runs": n_runs,
                    "cases": case_summaries, "traces": all_traces},
                   default=str, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("OVERALL SUMMARY  [tool-dispatch regression]")
    print(f"  MUST pass rate:    {total_must_pass}/{total_must_total} "
          f"({total_must_pass/total_must_total:.0%})")
    print(f"  SHOULD pass rate:  {total_should_pass}/{total_should_total} "
          f"({total_should_pass/total_should_total:.0%})")
    print(f"  Mean report quality: {total_quality_pts/total_quality_max:.0%} "
          f"({total_quality_pts}/{total_quality_max})")
    print(f"\nFull traces saved to: {out_path}")

    must_failures = [s for s in case_summaries if s["must"] != f"{n_runs}/{n_runs}"]
    if must_failures:
        print("\n*** MUST FAILURES (real bugs) ***")
        for s in must_failures:
            print(f"  {s['id']}: {s['must']}")
            if s["first_partial"]:
                print(f"    {json.dumps(s['first_partial'], default=str)[:300]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of agent runs per case (default 5)")
    args = parser.parse_args()
    run_eval_dispatch(args.runs)
