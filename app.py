import os
import sqlite3
import json
import hashlib
import secrets
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent import MODEL, run_agent

st.set_page_config(page_title="Research & Competitor Intelligence Agent", page_icon="🛰️", layout="wide")

DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            trace TEXT NOT NULL,
            search_queries TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()


def authenticate(username: str, password: str) -> tuple[bool, str]:
    conn = _get_conn()
    row = conn.execute("SELECT salt, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        salt = secrets.token_hex(16)
        password_hash = _hash_password(password, salt)
        conn.execute("INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)", (username, salt, password_hash))
        conn.commit()
        conn.close()
        return True, "Account created and logged in."
    salt, stored_hash = row
    conn.close()
    if _hash_password(password, salt) == stored_hash:
        return True, "Logged in."
    return False, "Incorrect password for that username."


def save_session(username: str, query: str, result: dict):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (username, query, answer, trace, search_queries, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            username,
            query,
            result.get("answer", ""),
            json.dumps(result.get("trace", [])),
            json.dumps(result.get("search_queries", [])),
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.commit()
    conn.close()


def get_history(username: str):
    conn = _get_conn()
    rows = conn.execute("SELECT id, query, created_at FROM sessions WHERE username = ? ORDER BY id DESC", (username,)).fetchall()
    conn.close()
    return rows


def get_session(session_id: int):
    conn = _get_conn()
    row = conn.execute("SELECT query, answer, trace, search_queries, created_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    query, answer, trace_json, search_queries_json, created_at = row
    return {
        "query": query,
        "answer": answer,
        "trace": json.loads(trace_json),
        "search_queries": json.loads(search_queries_json),
        "created_at": created_at,
    }


def build_long_term_context(username: str, limit: int = 3) -> str:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT query, answer, created_at FROM sessions WHERE username = ? ORDER BY id DESC LIMIT ?",
        (username, limit),
    ).fetchall()
    conn.close()
    if not rows:
        return ""
    lines = []
    for query, answer, created_at in rows:
        summary = answer[:200].replace("\n", " ").strip()
        lines.append(f'- On {created_at}, researched "{query}": {summary}...')
    return "\n".join(lines)


def render_trace(trace: list):
    icons = {
        "context": "📚", "plan": "🗺️", "llm_call": "🤖", "tool_call": "🔧",
        "tool_failure": "⚠️", "tool_fallback": "🔁", "observation": "📄",
        "diagnosis": "🔎", "recovery_failure": "❌", "evaluator": "🧪",
        "analyst_final": "📊", "final": "🏁", "plan_error": "❌",
        "diagnosis_error": "⚠️", "evaluator_error": "⚠️", "analyst_error": "❌",
        "run_error": "❌",
    }
    labels = {
        "plan": "Planner Decision", "llm_call": "LLM Call", "tool_call": "Tool Call",
        "tool_failure": "Tool Failure", "tool_fallback": "Recovery / Fallback",
        "diagnosis": "Automatic Root-Cause Diagnosis", "observation": "Tool Observation",
        "evaluator": "Evaluator", "analyst_final": "Analyst Final", "final": "Run Complete",
        "context": "Run Context",
    }
    with st.expander(f"🧠 End-to-End Trace ({len(trace)} events)", expanded=True):
        for i, step in enumerate(trace, start=1):
            event_type = step.get("type", "event")
            icon = icons.get(event_type, "•")
            label = labels.get(event_type, event_type.replace("_", " ").title())
            status = step.get("status", "")
            st.markdown(f"**{i}. {icon} {label}** — `{status}`")
            cols = st.columns(4)
            cols[0].metric("Latency", f'{step.get("latency_ms", 0):.2f} ms')
            cols[1].metric("Input tokens", step.get("input_tokens", 0))
            cols[2].metric("Output tokens", step.get("output_tokens", 0))
            cols[3].metric("Total tokens", step.get("total_tokens", 0))
            if step.get("tool"):
                st.caption(f'Tool: `{step.get("tool")}` | Query: `{step.get("query", "")}`')
            if step.get("error"):
                st.error(step["error"])
            content = step.get("content", "")
            if content:
                st.code(str(content))


def render_metrics(result: dict):
    trace = result.get("trace", [])
    llm_tokens = sum(int(t.get("total_tokens", 0) or 0) for t in trace if t.get("type") == "llm_call")
    tool_calls = len([t for t in trace if t.get("type") == "tool_call"])
    failures = len([t for t in trace if t.get("type") == "tool_failure"])
    fallbacks = len([t for t in trace if t.get("type") == "tool_fallback"])
    st.subheader("📈 Run Telemetry")
    cols = st.columns(6)
    cols[0].metric("Latency", f'{result.get("latency_ms", 0):.0f} ms')
    cols[1].metric("LLM tokens", llm_tokens)
    cols[2].metric("Tool calls", tool_calls)
    cols[3].metric("Errors", failures)
    cols[4].metric("Recoveries", fallbacks)
    cols[5].metric("Success", "YES" if result.get("answer", "").strip() else "NO")

    if result.get("diagnosis"):
        st.info("🔎 **Automatic diagnosis:** " + result["diagnosis"])
    if result.get("recovery_applied"):
        st.success("🔁 Self-healing recovery was applied successfully.")


if "username" not in st.session_state:
    st.session_state.username = None
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

with st.sidebar:
    st.header("👤 Account")
    if st.session_state.username is None:
        name_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        st.caption("New username? An account is created automatically on first login.")
        if st.button("Log in / Sign up", use_container_width=True):
            if not name_input.strip() or not password_input:
                st.error("Enter both a username and a password.")
            else:
                success, message = authenticate(name_input.strip(), password_input)
                if success:
                    st.session_state.username = name_input.strip()
                    st.rerun()
                else:
                    st.error(message)
    else:
        st.success(f"Logged in as **{st.session_state.username}**")
        if st.button("Log out", use_container_width=True):
            st.session_state.username = None
            st.session_state.active_session_id = None
            st.rerun()

        st.divider()
        st.header("🕘 Search History")
        history_rows = get_history(st.session_state.username)
        if not history_rows:
            st.caption("No past searches yet.")
        else:
            for session_id, query, created_at in history_rows:
                label = f"{query[:40]}{'...' if len(query) > 40 else ''}"
                if st.button(label, key=f"hist_{session_id}", use_container_width=True, help=created_at):
                    st.session_state.active_session_id = session_id
                    st.rerun()
        if st.button("➕ New search", use_container_width=True):
            st.session_state.active_session_id = None
            st.rerun()

        st.divider()
        st.header("🧪 Controlled Failure")
        st.checkbox(
            "Force tool failures this run",
            key="simulate_failure_toggle",
            help="The primary tool attempt is intentionally failed. The diagnostic agent identifies the root cause and selects a fallback tool.",
        )
        if st.session_state.get("simulate_failure_toggle"):
            st.warning("Demo mode ON — primary tool calls will fail intentionally.")

st.title("🛰️ Research & Competitor Intelligence Agent")
st.caption(
    "LangGraph research agent with end-to-end observability: planner decisions, prompts/LLM calls, "
    "tool calls, latency, token usage, controlled failures, automatic diagnosis, recovery, and evaluation."
)

if st.session_state.username is None:
    st.info("Log in from the sidebar to start searching and save history.")
    st.stop()

if st.session_state.active_session_id is not None:
    session_data = get_session(st.session_state.active_session_id)
    if session_data:
        st.subheader(f'Search: *{session_data["query"]}*')
        st.caption(f'Executed on {session_data["created_at"]}')
        render_trace(session_data["trace"])
        st.markdown("---")
        st.markdown(session_data["answer"])
else:
    user_input = st.text_area("What topic or competitor do you want to investigate?", height=100)
    if st.button("Run Intelligence Agent", type="primary", use_container_width=True):
        if not user_input.strip():
            st.error("Please enter a valid search topic.")
        else:
            with st.spinner("Running planner → tools → diagnosis → recovery → evaluator → analyst..."):
                past_context = build_long_term_context(st.session_state.username)
                result = run_agent(
                    user_input.strip(),
                    long_term_context=past_context,
                    simulate_failure=st.session_state.get("simulate_failure_toggle", False),
                )
            save_session(st.session_state.username, user_input.strip(), result)
            render_metrics(result)
            render_trace(result.get("trace", []))
            st.markdown("---")
            if result.get("answer"):
                st.subheader("📊 Final Briefing")
                st.markdown(result["answer"])
            else:
                st.error("The agent did not produce a final answer. Check the trace above for the root cause.")