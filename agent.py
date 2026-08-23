


import os
import json
import operator
import time
import uuid
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, Annotated, List, Dict, Any

from google import genai
from google.genai import types
from ddgs import DDGS
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    import streamlit as st
    _API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
except Exception:
    _API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_ROUNDS = 3
FALLBACK_TOOL = {"search_web": "search_arxiv", "search_arxiv": "search_web"}

if not _API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured. Put it in .env or Streamlit secrets.")

client = genai.Client(api_key=_API_KEY)


def _usage_value(usage: Any, *names: str) -> int:
    if usage is None:
        return 0
    for name in names:
        if isinstance(usage, dict) and name in usage:
            try:
                return int(usage[name] or 0)
            except Exception:
                pass
        value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value or 0)
            except Exception:
                pass
    return 0


def _telemetry_entry(
    *,
    run_id: str,
    agent: str,
    operation: str,
    start: float,
    status: str,
    content: str = "",
    usage: Any = None,
    error: str = "",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    input_tokens = _usage_value(usage, "prompt_token_count", "input_token_count")
    output_tokens = _usage_value(usage, "candidates_token_count", "output_token_count")
    total_tokens = _usage_value(usage, "total_token_count") or (input_tokens + output_tokens)
    item = {
        "type": operation,
        "agent": agent,
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "status": status,
        "content": content,
    }
    if error:
        item["error"] = error
    if extra:
        item.update(extra)
    return item


def _generate(
    *,
    run_id: str,
    agent_name: str,
    prompt: str,
    system_instruction: str,
    schema: types.Schema | None = None,
) -> tuple[Any, Dict[str, Any]]:
    start = time.perf_counter()
    config_kwargs: Dict[str, Any] = {"system_instruction": system_instruction}
    if schema is not None:
        config_kwargs.update({"response_mime_type": "application/json", "response_schema": schema})
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        usage = getattr(response, "usage_metadata", None)
        telemetry = _telemetry_entry(
            run_id=run_id,
            agent=agent_name,
            operation="llm_call",
            start=start,
            status="success",
            usage=usage,
            extra={"model": MODEL},
        )
        return response, telemetry
    except Exception as exc:
        telemetry = _telemetry_entry(
            run_id=run_id,
            agent=agent_name,
            operation="llm_call",
            start=start,
            status="error",
            error=str(exc),
            extra={"model": MODEL},
        )
        raise RuntimeError(json.dumps(telemetry)) from exc


# -----------------------------------------------------------------------------
# TOOLS
# -----------------------------------------------------------------------------

def _execute_web_search(query: str, max_results: int = 5) -> str:
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No web results found for this query."
    lines = []
    for r in results:
        title = str(r.get("title", "")).strip()
        body = str(r.get("body", "")).strip()
        href = str(r.get("href", "")).strip()
        lines.append(f"- {title}: {body} (source: {href})")
    return "\n".join(lines)


_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _execute_arxiv_search(query: str, max_results: int = 5) -> str:
    params = urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    entries = root.findall("atom:entry", _ARXIV_NS)
    if not entries:
        return "No arXiv papers found for this query."
    lines = []
    for entry in entries:
        title = (entry.findtext("atom:title", default="", namespaces=_ARXIV_NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS) or "").strip().replace("\n", " ")
        link = (entry.findtext("atom:id", default="", namespaces=_ARXIV_NS) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=_ARXIV_NS) or "").strip()[:10]
        snippet = summary[:280] + ("..." if len(summary) > 280 else "")
        lines.append(f"- {title} ({published}): {snippet} (source: {link})")
    return "\n".join(lines)


_TOOL_EXECUTORS = {"search_web": _execute_web_search, "search_arxiv": _execute_arxiv_search}


def _run_primary_tool(tool: str, query: str, simulate_failure: bool, run_id: str) -> tuple[str, Dict[str, Any]]:
    start = time.perf_counter()
    if simulate_failure:
        exc = RuntimeError("Simulated transient tool failure (adversarial mode)")
        trace = _telemetry_entry(
            run_id=run_id,
            agent="Tool Executor",
            operation="tool_failure",
            start=start,
            status="error",
            content=f'{tool}("{query}") failed intentionally.',
            error=str(exc),
            extra={"tool": tool, "query": query},
        )
        return "", trace
    try:
        result = _TOOL_EXECUTORS[tool](query)
        trace = _telemetry_entry(
            run_id=run_id,
            agent="Tool Executor",
            operation="tool_call",
            start=start,
            status="success",
            content=result,
            extra={"tool": tool, "query": query},
        )
        return result, trace
    except Exception as exc:
        trace = _telemetry_entry(
            run_id=run_id,
            agent="Tool Executor",
            operation="tool_failure",
            start=start,
            status="error",
            content=f'{tool}("{query}") failed.',
            error=str(exc),
            extra={"tool": tool, "query": query},
        )
        return "", trace


# -----------------------------------------------------------------------------
# STATE
# -----------------------------------------------------------------------------

class ToolCallSpec(TypedDict):
    tool: str
    query: str


class AgentState(TypedDict, total=False):
    user_input: str
    long_term_context: str
    findings: Annotated[List[str], operator.add]
    trace: Annotated[List[Dict[str, Any]], operator.add]
    tried_actions: Annotated[List[str], operator.add]
    round_count: int
    plan: List[ToolCallSpec]
    need_more_research: bool
    conflict_notes: str
    final_report: str
    simulate_failure: bool
    run_id: str
    diagnosis: str
    recovery_tool: str
    recovery_query: str
    recovery_applied: bool


# -----------------------------------------------------------------------------
# SCHEMAS
# -----------------------------------------------------------------------------

_PLAN_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "need_more_research": types.Schema(type=types.Type.BOOLEAN),
        "reasoning": types.Schema(type=types.Type.STRING),
        "tool_calls": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "tool": types.Schema(type=types.Type.STRING, enum=["search_web", "search_arxiv"]),
                    "query": types.Schema(type=types.Type.STRING),
                },
                required=["tool", "query"],
            ),
            max_items=2,
        ),
    },
    required=["need_more_research", "reasoning", "tool_calls"],
)

_EVAL_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "sufficient": types.Schema(type=types.Type.BOOLEAN),
        "reasoning": types.Schema(type=types.Type.STRING),
        "conflicts_found": types.Schema(type=types.Type.BOOLEAN),
        "conflict_notes": types.Schema(type=types.Type.STRING),
    },
    required=["sufficient", "reasoning", "conflicts_found", "conflict_notes"],
)

_DIAG_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "failure_detected": types.Schema(type=types.Type.BOOLEAN),
        "root_cause": types.Schema(type=types.Type.STRING),
        "recovery_tool": types.Schema(type=types.Type.STRING, enum=["search_web", "search_arxiv", "none"]),
        "improvement": types.Schema(type=types.Type.STRING),
    },
    required=["failure_detected", "root_cause", "recovery_tool", "improvement"],
)


# -----------------------------------------------------------------------------
# NODES
# -----------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    run_id = state.get("run_id", str(uuid.uuid4()))
    round_no = state.get("round_count", 0) + 1
    findings = "\n".join(state.get("findings", [])) or "(none yet)"
    tried = "\n".join(state.get("tried_actions", [])) or "(none yet)"
    diagnosis = state.get("diagnosis", "None")

    prompt = (
        f"User request: {state.get('user_input', '')}\n\n"
        f"Long-term context:\n{state.get('long_term_context', '') or '(none)'}\n\n"
        f"Findings:\n{findings}\n\n"
        f"Actions already tried:\n{tried}\n\n"
        f"Previous failure diagnosis/improvement:\n{diagnosis}\n\n"
        "Plan the next research step. Choose 1-2 tool calls. Do not repeat an action already tried "
        "unless the previous attempt failed and there is a clear reason to retry."
    )
    try:
        response, telemetry = _generate(
            run_id=run_id,
            agent_name="Planner",
            prompt=prompt,
            system_instruction="You are the Planning module. Make concise, evidence-oriented tool decisions.",
            schema=_PLAN_SCHEMA,
        )
        data = json.loads(response.text)
        plan = data.get("tool_calls", [])[:2]
        content = f"Round {round_no}: {data.get('reasoning', '')}"
        telemetry.update({"type": "plan", "content": content, "round": round_no, "decision": plan})
        return {
            "plan": plan,
            "need_more_research": bool(data.get("need_more_research")) and bool(plan),
            "round_count": round_no,
            "trace": [telemetry],
        }
    except Exception as exc:
        return {
            "plan": [],
            "need_more_research": False,
            "round_count": round_no,
            "trace": [{
                "type": "plan_error", "agent": "Planner", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "latency_ms": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "error", "content": "Planner failed; stopping safely.", "error": str(exc),
            }],
        }


def execute_tools_node(state: AgentState) -> dict:
    run_id = state.get("run_id", str(uuid.uuid4()))
    plan = state.get("plan", [])
    simulate_failure = state.get("simulate_failure", False)
    if not plan:
        return {"findings": [], "trace": [], "tried_actions": []}

    all_trace: List[Dict[str, Any]] = []
    all_findings: List[str] = []
    tried: List[str] = []
    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        futures = [
            pool.submit(_run_primary_tool, call["tool"], call["query"], simulate_failure, run_id)
            for call in plan
        ]
        for call, future in zip(plan, futures):
            observation, entry = future.result()
            all_trace.append(entry)
            tried.append(f'{call["tool"]}("{call["query"]}")')
            if observation:
                all_findings.append(observation)
                all_trace.append({
                    "type": "observation", "agent": "Tool Executor", "run_id": run_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "latency_ms": 0,
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "status": "success", "content": observation,
                    "tool": call["tool"], "query": call["query"],
                })
    return {"findings": all_findings, "trace": all_trace, "tried_actions": tried}


def diagnostic_node(state: AgentState) -> dict:
    run_id = state.get("run_id", str(uuid.uuid4()))
    failures = [t for t in state.get("trace", []) if t.get("type") == "tool_failure"]
    if not failures:
        return {
            "diagnosis": "No tool failure detected. No recovery action required.",
            "recovery_tool": "",
            "recovery_query": "",
            "recovery_applied": False,
            "trace": [{
                "type": "diagnosis", "agent": "Diagnostic Agent", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "latency_ms": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "success", "content": "No failure detected; continuing normally.",
            }],
        }

    failure_summary = "\n".join(
        f"tool={x.get('tool')} query={x.get('query')} error={x.get('error', '')}" for x in failures
    )
    prompt = (
        f"User request: {state.get('user_input', '')}\n"
        f"Failures observed:\n{failure_summary}\n\n"
        "Diagnose the operational root cause. Choose the configured fallback tool for recovery when appropriate. "
        "Do not invent a new tool. Explain the improvement as a concise operational action."
    )
    try:
        response, telemetry = _generate(
            run_id=run_id,
            agent_name="Diagnostic Agent",
            prompt=prompt,
            system_instruction="You diagnose tool failures from telemetry and select safe recovery actions.",
            schema=_DIAG_SCHEMA,
        )
        data = json.loads(response.text)
        failed_tool = failures[0].get("tool", "")
        recovery_tool = data.get("recovery_tool", "none")
        if recovery_tool not in _TOOL_EXECUTORS:
            recovery_tool = FALLBACK_TOOL.get(failed_tool, "")
        recovery_query = failures[0].get("query", "")
        diagnosis = data.get("root_cause", "Unknown tool failure")
        improvement = data.get("improvement", "Use the fallback tool and record the failure pattern.")
        telemetry.update({
            "type": "diagnosis",
            "content": f"Root cause: {diagnosis}\nImprovement: {improvement}",
            "root_cause": diagnosis,
            "improvement": improvement,
            "recovery_tool": recovery_tool,
        })
        return {
            "diagnosis": f"{diagnosis} Improvement: {improvement}",
            "recovery_tool": recovery_tool,
            "recovery_query": recovery_query,
            "recovery_applied": False,
            "trace": [telemetry],
        }
    except Exception as exc:
        failed_tool = failures[0].get("tool", "")
        fallback = FALLBACK_TOOL.get(failed_tool, "")
        return {
            "diagnosis": f"Automatic diagnosis failed: {exc}. Deterministic fallback selected: {fallback}.",
            "recovery_tool": fallback,
            "recovery_query": failures[0].get("query", ""),
            "recovery_applied": False,
            "trace": [{
                "type": "diagnosis_error", "agent": "Diagnostic Agent", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "latency_ms": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "error", "content": "LLM diagnosis failed; deterministic fallback used.",
                "error": str(exc),
            }],
        }


def recovery_node(state: AgentState) -> dict:
    run_id = state.get("run_id", str(uuid.uuid4()))
    tool = state.get("recovery_tool", "")
    query = state.get("recovery_query", "")
    if not tool or tool not in _TOOL_EXECUTORS or not query:
        return {"recovery_applied": False, "trace": []}

    start = time.perf_counter()
    try:
        result = _TOOL_EXECUTORS[tool](query)
        entries = [
            {
                "type": "tool_fallback", "agent": "Recovery Agent", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "success",
                "content": f'Failure recovery active: {tool}("{query}")',
                "tool": tool, "query": query,
            },
            {
                "type": "observation", "agent": "Recovery Agent", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "latency_ms": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "success", "content": result,
                "tool": tool, "query": query,
            },
        ]
        return {"findings": [result], "recovery_applied": True, "trace": entries}
    except Exception as exc:
        return {
            "recovery_applied": False,
            "trace": [{
                "type": "recovery_failure", "agent": "Recovery Agent", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "error", "content": f"Recovery tool {tool} failed.", "error": str(exc),
            }],
        }


def evaluator_node(state: AgentState) -> dict:
    run_id = state.get("run_id", str(uuid.uuid4()))
    round_count = state.get("round_count", 0)
    tried_actions = state.get("tried_actions", [])
    deadlock = len(tried_actions) != len(set(tried_actions))
    if round_count >= MAX_ROUNDS or deadlock:
        reason = "Max loop safety cap reached" if round_count >= MAX_ROUNDS else "Repeated tool action detected"
        return {
            "need_more_research": False,
            "trace": [{
                "type": "evaluator", "agent": "Evaluator", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "latency_ms": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "success", "content": f"Forced termination: {reason}.",
            }],
        }

    findings = "\n".join(state.get("findings", [])) or "(none)"
    prompt = f"User request: {state.get('user_input', '')}\n\nFindings:\n{findings}\n\nEvaluate whether evidence is sufficient."
    try:
        response, telemetry = _generate(
            run_id=run_id,
            agent_name="Evaluator",
            prompt=prompt,
            system_instruction="You are the Evaluator. Check evidence sufficiency and conflicts.",
            schema=_EVAL_SCHEMA,
        )
        data = json.loads(response.text)
        telemetry.update({"type": "evaluator", "content": data.get("reasoning", "")})
        return {
            "need_more_research": not bool(data.get("sufficient", True)),
            "conflict_notes": data.get("conflict_notes", "") if data.get("conflicts_found") else state.get("conflict_notes", ""),
            "trace": [telemetry],
        }
    except Exception as exc:
        return {
            "need_more_research": False,
            "trace": [{
                "type": "evaluator_error", "agent": "Evaluator", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "latency_ms": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "error", "content": "Evaluator failed; proceeding safely.", "error": str(exc),
            }],
        }


def route_after_evaluator(state: AgentState) -> str:
    return "planner" if state.get("need_more_research") else "analyst"


def analyst_node(state: AgentState) -> dict:
    run_id = state.get("run_id", str(uuid.uuid4()))
    findings = "\n".join(state.get("findings", [])) or "(no findings)"
    prompt = (
        f"Original request: {state.get('user_input', '')}\n\n"
        f"Failure diagnosis / recovery: {state.get('diagnosis', 'None')}\n\n"
        f"Conflict notes: {state.get('conflict_notes', 'None')}\n\n"
        f"Collected evidence:\n{findings}\n\n"
        "Produce a concise research briefing. Clearly distinguish evidence from uncertainty and include source URLs from the evidence where available."
    )
    try:
        response, telemetry = _generate(
            run_id=run_id,
            agent_name="Analyst",
            prompt=prompt,
            system_instruction="You are the Analyst Agent. Synthesize evidence into a clear executive brief.",
        )
        final_report = (response.text or "").strip()
        telemetry.update({"type": "analyst_final", "content": final_report})
        return {"final_report": final_report, "trace": [telemetry]}
    except Exception as exc:
        return {
            "final_report": "The analyst step failed. See telemetry for the root cause.",
            "trace": [{
                "type": "analyst_error", "agent": "Analyst", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "latency_ms": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "error", "content": "Analyst failed.", "error": str(exc),
            }],
        }


# -----------------------------------------------------------------------------
# GRAPH + PUBLIC RUNNER
# -----------------------------------------------------------------------------

_builder = StateGraph(AgentState)
_builder.add_node("planner", planner_node)
_builder.add_node("execute_tools", execute_tools_node)
_builder.add_node("diagnostic", diagnostic_node)
_builder.add_node("recovery", recovery_node)
_builder.add_node("evaluator", evaluator_node)
_builder.add_node("analyst", analyst_node)

_builder.add_edge(START, "planner")
_builder.add_edge("planner", "execute_tools")
_builder.add_edge("execute_tools", "diagnostic")
_builder.add_edge("diagnostic", "recovery")
_builder.add_edge("recovery", "evaluator")
_builder.add_conditional_edges(
    "evaluator",
    route_after_evaluator,
    {"planner": "planner", "analyst": "analyst"},
)
_builder.add_edge("analyst", END)

graph = _builder.compile(checkpointer=MemorySaver())


def run_agent(user_input: str, long_term_context: str = "", simulate_failure: bool = False) -> dict:
    run_id = str(uuid.uuid4())
    initial_state: AgentState = {
        "user_input": user_input,
        "long_term_context": long_term_context,
        "findings": [],
        "trace": [],
        "tried_actions": [],
        "round_count": 0,
        "plan": [],
        "need_more_research": True,
        "conflict_notes": "",
        "final_report": "",
        "simulate_failure": simulate_failure,
        "run_id": run_id,
        "diagnosis": "",
        "recovery_tool": "",
        "recovery_query": "",
        "recovery_applied": False,
    }
    config = {"configurable": {"thread_id": run_id}}
    start = time.perf_counter()
    try:
        result = graph.invoke(initial_state, config=config)
        trace = result.get("trace", [])
        trace.insert(0, {
            "type": "context", "agent": "System", "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latency_ms": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "status": "success", "content": f"Run {run_id} started. Model: {MODEL}"
        })
        total_latency = round((time.perf_counter() - start) * 1000, 2)
        trace.append({
            "type": "final", "agent": "System", "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latency_ms": total_latency, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "status": "success", "content": f"Run completed in {total_latency:.2f} ms."
        })
        total_tokens = sum(int(t.get("total_tokens", 0) or 0) for t in trace)
        tool_calls = [t for t in trace if t.get("type") == "tool_call"]
        failures = [t for t in trace if t.get("type") == "tool_failure"]
        return {
            "answer": result.get("final_report", "").strip(),
            "trace": trace,
            "search_queries": result.get("tried_actions", []),
            "run_id": run_id,
            "latency_ms": total_latency,
            "total_tokens": total_tokens,
            "tool_calls": len(tool_calls),
            "tool_failures": len(failures),
            "tool_fallbacks": len([t for t in trace if t.get("type") == "tool_fallback"]),
            "diagnosis": result.get("diagnosis", ""),
            "recovery_applied": result.get("recovery_applied", False),
        }
    except Exception as exc:
        total_latency = round((time.perf_counter() - start) * 1000, 2)
        return {
            "answer": "",
            "trace": [{
                "type": "run_error", "agent": "System", "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "latency_ms": total_latency, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "status": "error", "content": "Agent run failed.", "error": str(exc),
            }],
            "search_queries": [], "run_id": run_id, "latency_ms": total_latency,
            "total_tokens": 0, "tool_calls": 0, "tool_failures": 0, "tool_fallbacks": 0,
            "diagnosis": "Run-level failure", "recovery_applied": False,
        }