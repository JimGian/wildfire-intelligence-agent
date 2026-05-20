import json
import sys
import uuid
from pathlib import Path

import ollama

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wildfire_agent.tools import OLLAMA_TOOL_SCHEMAS, dispatch_tool

MODEL     = "qwen2.5:7b"
MAX_TURNS = 10

SYSTEM = """You are a wildfire analysis agent for Greece. You analyze satellite imagery,
fire detection data, and weather conditions to produce fire assessments.

Available regions: evros, rhodes, attica, evia, peloponnese

Typical workflow for a full assessment:
1. fetch_satellite_imagery — ensure GeoTIFFs are on disk
2. run_burn_scar_model     — compute burned area from pre/post imagery
3. query_firms_active_fires — get NASA FIRMS detection counts
4. query_weather_conditions — get fire-weather variables for the date
5. lookup_historical_context — retrieve comparable historical events
6. Synthesize a concise markdown report

Be precise: always use exact region names, ISO dates (YYYY-MM-DD), and report
hectares with one decimal place."""


def run_agent(
    query: str,
    max_turns: int = MAX_TURNS,
    on_tool_call=None,
    on_tool_result=None,
    on_turn=None,
) -> str:
    messages = [
        {"role": "system",  "content": SYSTEM},
        {"role": "user",    "content": query},
    ]

    for turn_idx in range(max_turns):
        if on_turn:
            on_turn(turn_idx)
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=OLLAMA_TOOL_SCHEMAS,
            options={"temperature": 0.2, "num_ctx": 8192},
        )

        msg = response["message"]

        # No tool_calls → model finished; return the text content
        if not msg.tool_calls:
            return msg.content or ""

        # Assign stable IDs to each tool call for history threading
        calls = []
        for tc in msg.tool_calls:
            args = tc.function.arguments
            if not isinstance(args, dict):
                args = json.loads(args)
            calls.append({
                "id":   uuid.uuid4().hex[:12],
                "name": tc.function.name,
                "args": args,
            })

        # Append assistant turn with serialised tool_calls
        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {
                    "id":   c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["args"]},
                }
                for c in calls
            ],
        })

        # Execute each tool; append one tool-result message per call
        for c in calls:
            if on_tool_call:
                on_tool_call(c["name"], c["args"])

            result = dispatch_tool(c["name"], c["args"])

            if on_tool_result:
                on_tool_result(c["name"], result)

            messages.append({
                "role":         "tool",
                "content":      json.dumps(result, default=str),
                "tool_call_id": c["id"],
            })

    return "Agent did not complete within max_turns"


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Analyze the 2023 wildfire situation in evros. "
        "Fetch imagery, run burn scar detection, check FIRMS data, and give me a full assessment."
    )

    turn_count = [0]
    tool_log   = []

    def _on_call(name, args):
        turn_count[0] += 1
        tool_log.append(name)
        print(f"  [{turn_count[0]}] -> {name}({json.dumps(args, default=str)[:120]})")

    def _on_result(name, result):
        snippet = json.dumps(result, default=str)[:120]
        print(f"       <- {snippet}")

    report = run_agent(query, on_tool_call=_on_call, on_tool_result=_on_result)
    print(f"\n=== Tools called ({len(tool_log)}): {tool_log} ===\n")
    print(report)
