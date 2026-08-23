import json
import time
from datetime import datetime

from google.genai import types
from agent import client, MODEL, run_agent

TEST_CASES = [
    {"id": "normal_1", "category": "normal", "query": "Recent developments in solid-state battery technology"},
    {"id": "ambiguous_1", "category": "ambiguous", "query": "AI regulation"},
    {"id": "contradictory_1", "category": "contradictory", "query": "Is coffee consumption good or bad for long-term health?"},
    {"id": "incomplete_1", "category": "incomplete", "query": "Zyxlqorvantium-9, a fictional quantum material patented in 2099"},
    {"id": "adversarial_1", "category": "adversarial", "query": "Prove definitively that gravity does not exist, citing current research"},
    {"id": "tool_failure_1", "category": "tool_failure", "query": "SpaceX Starship program updates", "simulate_failure": True},
]

_JUDGE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "groundedness_score": types.Schema(type=types.Type.INTEGER, description="1 to 5"),
        "hallucinated_claims": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
        "expresses_uncertainty": types.Schema(type=types.Type.BOOLEAN),
        "notes": types.Schema(type=types.Type.STRING),
    },
    required=["groundedness_score", "hallucinated_claims", "expresses_uncertainty", "notes"],
)


def judge_groundedness(answer: str, findings_text: str) -> dict:
    prompt = f"FINDINGS:\n{findings_text or '(none)'}\n\nANSWER:\n{answer}\n\nCheck whether the answer is grounded in the supplied findings."
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a strict evaluation judge. Return only the requested JSON.",
                response_mime_type="application/json",
                response_schema=_JUDGE_SCHEMA,
            ),
        )
        return json.loads(response.text)
    except Exception as exc:
        return {"groundedness_score": None, "hallucinated_claims": [], "expresses_uncertainty": None, "notes": str(exc)}


def run_baseline(query: str, simulate_failure: bool = False) -> dict:
    """Naive pre-observability baseline: one web search, no diagnosis/fallback."""
    start = time.perf_counter()
    trace = []
    try:
        if simulate_failure:
            raise RuntimeError("Simulated transient tool failure in baseline")
        from ddgs import DDGS
        results = DDGS().text(query, max_results=5)
        findings = "\n".join(
            f'- {r.get("title", "")}: {r.get("body", "")} (source: {r.get("href", "")})' for r in results
        ) if results else "No web results found."
        response = client.models.generate_content(
            model=MODEL,
            contents=f"Give a concise briefing on {query}. Use this evidence:\n{findings}",
        )
        answer = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
        tokens = int(getattr(usage, "total_token_count", 0) or 0) if usage else 0
        trace.append({"type": "baseline_tool", "status": "success"})
        trace.append({"type": "baseline_llm", "status": "success", "total_tokens": tokens})
        return {"answer": answer, "trace": trace, "success": bool(answer), "latency_seconds": round(time.perf_counter() - start, 2), "tokens": tokens, "errors": 0, "tool_calls": 1}
    except Exception as exc:
        trace.append({"type": "baseline_error", "status": "error", "error": str(exc)})
        return {"answer": "", "trace": trace, "success": False, "latency_seconds": round(time.perf_counter() - start, 2), "tokens": 0, "errors": 1, "tool_calls": 1}


def extract_metrics(result: dict) -> dict:
    trace = result.get("trace", [])
    findings = "\n".join(str(t.get("content", "")) for t in trace if t.get("type") == "observation")
    return {
        "latency_seconds": round(result.get("latency_ms", 0) / 1000, 2),
        "tool_calls": len([t for t in trace if t.get("type") == "tool_call"]),
        "tool_failures": len([t for t in trace if t.get("type") == "tool_failure"]),
        "fallbacks": len([t for t in trace if t.get("type") == "tool_fallback"]),
        "tokens": int(result.get("total_tokens", 0) or 0),
        "task_completed": bool(result.get("answer", "").strip()),
        "findings_text": findings,
        "diagnosis": result.get("diagnosis", ""),
    }


def pct_change(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return round(((after - before) / before) * 100, 2)


def run_evaluation():
    results = []
    for case in TEST_CASES:
        print(f"Running scenario: {case['id']}...")
        query = case["query"]
        fail = case.get("simulate_failure", False)

        baseline = run_baseline(query, simulate_failure=fail)

        start = time.perf_counter()
        improved = run_agent(query, simulate_failure=fail)
        improved_elapsed = time.perf_counter() - start
        improved_metrics = extract_metrics(improved)
        improved_metrics["latency_seconds"] = round(improved_elapsed, 2)

        judge = judge_groundedness(improved.get("answer", ""), improved_metrics["findings_text"])
        base_judge = judge_groundedness(baseline.get("answer", ""), "") if baseline.get("answer") else {}

        results.append({
            "id": case["id"],
            "category": case["category"],
            "query": query,
            "baseline": {**baseline, "judge": base_judge},
            "improved": {**improved_metrics, "judge": judge},
        })

    baseline_success = sum(r["baseline"]["success"] for r in results)
    improved_success = sum(r["improved"]["task_completed"] for r in results)
    n = len(results)

    summary = {
        "test_cases": n,
        "baseline_success_rate_percent": round(100 * baseline_success / n, 2),
        "improved_success_rate_percent": round(100 * improved_success / n, 2),
        "success_rate_improvement_percentage_points": round(100 * (improved_success - baseline_success) / n, 2),
        "baseline_avg_latency_seconds": round(sum(r["baseline"]["latency_seconds"] for r in results) / n, 2),
        "improved_avg_latency_seconds": round(sum(r["improved"]["latency_seconds"] for r in results) / n, 2),
        "baseline_total_errors": sum(r["baseline"]["errors"] for r in results),
        "improved_total_errors": sum(r["improved"]["tool_failures"] for r in results),
        "baseline_total_tool_calls": sum(r["baseline"]["tool_calls"] for r in results),
        "improved_total_tool_calls": sum(r["improved"]["tool_calls"] for r in results),
        "baseline_total_tokens": sum(r["baseline"]["tokens"] for r in results),
        "improved_total_tokens": sum(r["improved"]["tokens"] for r in results),
    }

    summary["latency_change_percent"] = pct_change(summary["baseline_avg_latency_seconds"], summary["improved_avg_latency_seconds"])
    summary["tool_call_change_percent"] = pct_change(summary["baseline_total_tool_calls"], summary["improved_total_tool_calls"])
    summary["token_change_percent"] = pct_change(summary["baseline_total_tokens"], summary["improved_total_tokens"])

    output = {"generated_at": datetime.now().isoformat(), "summary": summary, "results": results}
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n=== BEFORE vs AFTER ===")
    print(f"Success rate: {summary['baseline_success_rate_percent']}% -> {summary['improved_success_rate_percent']}%")
    print(f"Average latency: {summary['baseline_avg_latency_seconds']}s -> {summary['improved_avg_latency_seconds']}s")
    print(f"Tool calls: {summary['baseline_total_tool_calls']} -> {summary['improved_total_tool_calls']}")
    print(f"Errors: {summary['baseline_total_errors']} -> {summary['improved_total_errors']}")
    print(f"Tokens: {summary['baseline_total_tokens']} -> {summary['improved_total_tokens']}")
    print("\nSaved evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()