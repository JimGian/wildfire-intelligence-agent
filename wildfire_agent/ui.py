"""
Gradio streaming UI for the wildfire analysis agent.

Run:  python wildfire_agent/ui.py
URL:  http://127.0.0.1:7861
"""
import html as _html
import json
import sys
import time
from pathlib import Path

import uuid

import gradio as gr
import ollama  # local Ollama backend — no API key needed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wildfire_agent.tools import OLLAMA_TOOL_SCHEMAS, dispatch_tool
from wildfire_agent.agent import MODEL, SYSTEM  # keeps model/system in one place

MAX_TURNS = 12

# (chip label shown in UI, full query text sent to agent)
EXAMPLES = [
    (
        "Evros 2023 full assessment",
        "Analyze the 2023 Evros wildfire. Fetch imagery, run burn scar detection, "
        "check FIRMS data starting Aug 19 2023 for 30 days, and give me a full assessment.",
    ),
    (
        "Rhodes 2023 full assessment",
        "Run a full assessment of the 2023 Rhodes wildfire.",
    ),
    (
        "Compare Mati 2018 vs Evros 2023",
        "Compare the Mati 2018 fire to the Evros 2023 fire — what made Mati so deadly "
        "despite being much smaller in area?",
    ),
    (
        "Dadia fire weather conditions",
        "What weather conditions preceded the Dadia fire on August 19 2023? "
        "How do they compare to typical fire-weather thresholds?",
    ),
    (
        "2023 season severity vs history",
        "Is the 2023 Greek fire season unusually severe by historical standards?",
    ),
]


# ─── One-line result summaries ───────────────────────────────────────────────

def _summarize(name: str, result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"
    if name == "run_burn_scar_model":
        ha   = result.get("burned_ha", {}).get("moderate_0.27", "?")
        fp   = result.get("fire_probability", "?")
        pct  = result.get("burned_pct_of_scene", "?")
        ha_s = f"{ha:,.1f}" if isinstance(ha, (int, float)) else str(ha)
        fp_s = f"{fp:.2f}" if isinstance(fp, float) else str(fp)
        return f"→ {ha_s} ha at moderate+ severity ({pct}% of scene); fire prob {fp_s}"
    if name == "fetch_satellite_imagery":
        periods = [k for k in result if k in ("pre_fire", "during", "post_fire")]
        missing = result.get("missing", [])
        if missing:
            return f"→ {len(periods)} TIFFs found; missing: {', '.join(missing)}"
        return f"→ {len(periods)} TIFFs cached ({', '.join(periods)})"
    if name == "query_firms_active_fires":
        n    = result.get("detection_count", 0)
        peak = result.get("peak_day")
        pk_n = result.get("peak_day_count", 0)
        days = result.get("window_covered_days", "?")
        if n == 0:
            return f"→ 0 detections in {days}-day window"
        pk_s = f"; peak {peak} ({pk_n:,})" if peak else ""
        return f"→ {n:,} detections in {days}-day window{pk_s}"
    if name == "query_weather_conditions":
        t   = result.get("temp_max", "?")
        w   = result.get("wind_speed_max_kmh", "?")
        rh  = result.get("relative_humidity_mean_pct", "?")
        idx = result.get("hot_dry_windy_index", "?")
        return f"→ {t}°C max, {w} km/h wind, {rh}% RH — hot-dry-windy index {idx}"
    if name == "lookup_historical_context":
        top = (result.get("results") or [{}])[0]
        if top.get("title"):
            return f"→ top: '{top['title']}' (sim {top.get('similarity', 0):.4f})"
        return "→ no results"
    return f"→ {json.dumps(result, default=str)[:120]}"


# ─── Live log renderer ───────────────────────────────────────────────────────

_PLACEHOLDER = (
    "<p style='color:#aaa; font-style:italic; padding:8px 4px;'>"
    "Agent activity will stream here as the query runs."
    "</p>"
)


def _render_log(events: list) -> str:
    if not events:
        return _PLACEHOLDER
    parts = []
    for e in events:
        pending     = e["status"] == "pending"
        icon        = "&#9203;" if pending else "&#9989;"   # ⏳ / ✅
        border_col  = "#bbb"    if pending else "#4caf50"
        args_html   = _html.escape(json.dumps(e["args"], indent=2))

        block = (
            f'<div style="margin:6px 0; padding:10px 14px; '
            f'border-left:3px solid {border_col}; background:#fafafa; '
            f'font-family:monospace; font-size:13px; border-radius:0 4px 4px 0;">'
            f'<b>{icon} {_html.escape(e["name"])}</b>'
            f'<details style="margin-top:5px;">'
            f'<summary style="cursor:pointer; color:#aaa; font-size:11px; '
            f'user-select:none;">args</summary>'
            f'<pre style="margin:4px 0; font-size:11px; background:#ebebeb; '
            f'padding:6px; border-radius:3px; overflow:auto; '
            f'max-height:140px;">{args_html}</pre>'
            f'</details>'
        )
        if not pending:
            summary_h   = _html.escape(e.get("summary", ""))
            result_json = _html.escape(json.dumps(e["result"], indent=2, default=str))
            block += (
                f'<div style="color:#2a7a2a; font-size:12px; margin-top:6px;">'
                f'{summary_h}</div>'
                f'<details style="margin-top:3px;">'
                f'<summary style="cursor:pointer; color:#aaa; font-size:11px; '
                f'user-select:none;">full result</summary>'
                f'<pre style="margin:4px 0; font-size:11px; background:#ebebeb; '
                f'padding:6px; border-radius:3px; max-height:200px; '
                f'overflow:auto;">{result_json}</pre>'
                f'</details>'
            )
        block += "</div>"
        parts.append(block)
    return "\n".join(parts)


# ─── Streaming agent loop ────────────────────────────────────────────────────

def run_query(query: str):
    """Generator: yields (log_html, report_md) after each tool call and at completion."""
    if not query or not query.strip():
        yield _PLACEHOLDER, ""
        return

    events: list     = []
    tool_order: list = []
    start            = time.time()
    report_md        = ""

    yield _render_log(events), ""

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": query.strip()},
    ]

    for _turn in range(MAX_TURNS):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=OLLAMA_TOOL_SCHEMAS,
                options={"temperature": 0.2, "num_ctx": 16384},
            )
        except Exception as exc:
            yield _render_log(events), f"**Ollama error:** `{exc}`"
            return

        msg = response["message"]

        # No tool_calls → model is done; extract report
        if not msg.tool_calls:
            report_md = msg.content or ""
            elapsed = time.time() - start
            if tool_order:
                calls_li = "\n".join(
                    f"{i + 1}. `{n}`" for i, n in enumerate(tool_order)
                )
                report_md += (
                    "\n\n---\n"
                    "<details>\n<summary>How this was generated</summary>\n\n"
                    f"**{len(tool_order)} tool call(s)** — {elapsed:.1f}s wall-clock\n\n"
                    f"{calls_li}\n\n"
                    "</details>"
                )
            yield _render_log(events), report_md
            return

        # Assign stable IDs; build and append assistant history entry
        calls = []
        for tc in msg.tool_calls:
            args = tc.function.arguments
            if not isinstance(args, dict):
                args = json.loads(args)
            calls.append({"id": uuid.uuid4().hex[:12], "name": tc.function.name, "args": args})

        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"]}}
                for c in calls
            ],
        })

        # Execute tools; stream ⏳ → ✅ for each
        for c in calls:
            ev = {"name": c["name"], "args": c["args"],
                  "status": "pending", "result": {}, "summary": ""}
            events.append(ev)
            tool_order.append(c["name"])
            yield _render_log(events), report_md   # ⏳

            result = dispatch_tool(c["name"], c["args"])

            ev["status"]  = "done"
            ev["result"]  = result
            ev["summary"] = _summarize(c["name"], result)
            yield _render_log(events), report_md   # ✅

            messages.append({
                "role":         "tool",
                "content":      json.dumps(result, default=str),
                "tool_call_id": c["id"],
            })

    # Exhausted MAX_TURNS without end_turn
    elapsed = time.time() - start
    note = (
        f"\n\n---\n*Agent did not complete within {MAX_TURNS} turns "
        f"({elapsed:.1f}s). Partial trace visible in the activity panel.*"
    )
    yield _render_log(events), (report_md or "") + note


# ─── UI layout ───────────────────────────────────────────────────────────────

with gr.Blocks(title="Wildfire Intelligence Agent (local Qwen 2.5 7B)") as demo:
    with gr.Row(equal_height=False):

        # ── Left column: inputs ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            query_box = gr.Textbox(
                label="Query",
                placeholder="Ask about a wildfire event or region...",
                lines=5,
            )
            gr.Markdown("**Example queries**")
            # Two rows of chip buttons; each populates the textbox when clicked
            for i in range(0, len(EXAMPLES), 3):
                with gr.Row():
                    for label, full_text in EXAMPLES[i : i + 3]:
                        chip = gr.Button(label, size="sm")
                        chip.click(fn=lambda t=full_text: t, outputs=query_box)
            run_btn = gr.Button("Run analysis", variant="primary")

        # ── Right column: live log + report ──────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Agent activity")
            log_out = gr.HTML(value=_PLACEHOLDER)
            gr.Markdown("### Report")
            report_out = gr.Markdown(value="")

    run_btn.click(
        fn=run_query,
        inputs=[query_box],
        outputs=[log_out, report_out],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
    )
