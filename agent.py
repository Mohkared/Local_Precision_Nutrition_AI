"""
agent.py
────────────────────────────────────────────────────────────────────────────
Enhanced ReAct (Reason + Act) agent for Precision Nutrition AI.

Fixes applied vs original:
  • Safety pre-screening via safety.py (short-circuits critical queries)
  • Real RAG retrieval with citation extraction
  • Structured result with citations, safety flags, tool usage stats
  • 4 modes: baseline | rag_only | agent_only | agent+rag (full)
  • Tool timeouts to prevent hanging on long calls
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
from typing import Callable, Optional

import ollama

from tools import AGENT_TOOLS, RAG_TOOL, TOOL_FUNCTIONS
from rag_engine import retrieve_as_string
from safety import screen_message, format_safety_disclaimer


# ── Model configuration ────────────────────────────────────────────────────
MODEL_NAME           = "qwen3.5:4b"   # Change to your local model; must support tool calling
MAX_STEPS            = 5              # Hard cap on ReAct iterations
TOOL_TIMEOUT_SECONDS = 30             # FIX: max seconds per tool call before aborting

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Precision Nutrition AI — a knowledgeable, evidence-based dietary counsellor.

CORE PRINCIPLES:
1. Always ground responses in USDA, ADA, AHA, NIH, or WHO guidelines.
2. When RAG context is provided, cite sources using their citation labels [1], [2], etc.
3. When you have calculated values (BMR, TDEE, macros), USE them in your meal plan — do NOT contradict them.
4. Include a brief DISCLAIMER for any query involving a medical condition.
5. Never diagnose medical conditions or prescribe medication.
6. For sensitive cases (eating disorders, kidney disease, paediatric patients), refer to specialists — do NOT provide restrictive advice.

RESPONSE STRUCTURE (adapt length to query complexity):
• Understand: briefly acknowledge the user's situation (1–2 sentences)
• Calculate/Retrieve: use tools to get accurate numbers — show the key values
• Explain: connect recommendations to evidence-based guidelines (cite sources if RAG used)
• Actionable: provide specific foods, portions, and meal examples
• Engage: one closing question or offer to elaborate
• Disclaimer: required for any medical condition query
"""

# ── Semantic cache (in-process, session-scoped) ────────────────────────────
_cache: dict[str, dict] = {}

def _cache_key(message: str, use_agent: bool, use_rag: bool) -> str:
    raw = f"{message.strip().lower()}|{use_agent}|{use_rag}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Tool execution with timeout ────────────────────────────────────────────

def _call_tool_with_timeout(fn_name: str, fn_args: dict) -> str:
    """
    FIX: Execute a tool function inside a ThreadPoolExecutor and enforce
    TOOL_TIMEOUT_SECONDS.  Returns a safe error string if the call times out
    or raises an unexpected exception, so the agent can continue gracefully.
    """
    func = TOOL_FUNCTIONS.get(fn_name)
    if func is None:
        return f"ERROR: Unknown tool '{fn_name}'"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, **fn_args)
        try:
            return future.result(timeout=TOOL_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return (
                f"ERROR: Tool '{fn_name}' timed out after {TOOL_TIMEOUT_SECONDS} s. "
                "Please try a simpler query or continue without this tool result."
            )
        except TypeError as e:
            return f"ERROR: Bad arguments for '{fn_name}' — {e}"
        except Exception as e:
            return f"ERROR: Tool execution failed — {e}"


# ── Citation extraction from RAG results ──────────────────────────────────

# def _extract_citations(trace: list[str]) -> list[dict]:
#     """Parse citations from tool observations in the reasoning trace."""
#     citations = []
#     for entry in trace:
#         if "CITATIONS:" in entry:
#             parts = entry.split("CITATIONS:")[-1].strip().split(" | ")
#             for p in parts:
#                 p = p.strip()
#                 if p:
#                     citations.append({"label": p[:4] if p.startswith("[") else "?", "text": p})
#     return citations


# def _extract_and_store_citations(obs: str, citations: list) -> None:
#     """Parse citation lines from a RAG observation and add to citations list."""
#     if "CITATIONS:" not in obs:
#         return
#     print('Obs:', obs)
#     parts = obs.split("CITATIONS:")[-1].strip().split(" | ")
#     print('\n\n\nParts:', parts)
#     for p in parts:
#         p = p.strip()
#         if p and not any(c.get("text") == p for c in citations):
#             label = p.split()[0] if p.startswith("[") else "?"
#             citations.append({"label": label, "text": p, "content": ""})



def _extract_and_store_citations(obs: str, citations: list) -> None:
    """
    Parse a RAG tool observation and add each cited source to `citations`.
 
    Each citation dict contains:
        "label"   — short bracket label, e.g. "[1]"
        "text"    — full citation line from the CITATIONS: summary,
                    e.g. "[1] USDA Dietary Guidelines 2020-2025 (score: 0.85)"
        "content" — the actual retrieved passage for that label,
                    extracted from the RAG CONTEXT block above the CITATIONS line.
                    Empty string if the content block cannot be located.
 
    Observation format produced by retrieve_rag_context():
        RAG CONTEXT (cite these sources in your response):
        [1] [Source Name]
        <retrieved passage text>
        ---
        [2] [Another Source]
        <retrieved passage text>
 
        CITATIONS: [1] Source Name (score: 0.85) | [2] Another Source (score: 0.72)
    """
    if "CITATIONS:" not in obs:
        return
 
    # ── Step 1: build label → content map from the RAG CONTEXT block ─────
    # Split on "CITATIONS:" to isolate the content section.
    content_section = obs.split("CITATIONS:")[0]
 
    # Remove the preamble header line ("RAG CONTEXT …:")
    if "\n" in content_section:
        content_section = content_section[content_section.index("\n") + 1:]
 
    # Split on the "---" separators that divide individual chunks
    content_map: dict[str, str] = {}
    for block in content_section.split("---"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        # First line is the label+source header, e.g. "[1] [USDA Dietary Guidelines]"
        header = lines[0].strip() if lines else ""
        body   = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
 
        # Extract the bracket label "[N]" from the header
        if header.startswith("["):
            bracket_end = header.find("]")
            if bracket_end != -1:
                label_key = header[: bracket_end + 1]   # e.g. "[1]"
                content_map[label_key] = body
 
    # ── Step 2: parse the CITATIONS: summary line and build citation dicts ─
    citation_line = obs.split("CITATIONS:")[-1].strip()
    for part in citation_line.split(" | "):
        part = part.strip()
        if not part:
            continue
 
        # Deduplicate by text
        if any(c.get("text") == part for c in citations):
            continue
 
        label = part.split()[0] if part.startswith("[") else "?"
        citations.append({
            "label":   label,
            "text":    part,
            "content": content_map.get(label, ""),  # ← the exported retrieved text
        })
 
 


# ── Main agent function ────────────────────────────────────────────────────

def run_nutrition_agent(
    user_message: str,
    chat_history: Optional[list] = None,
    use_agent: bool = True,
    use_rag: bool = True,
    max_steps: int = MAX_STEPS,
    # should_stop: Callable[[], bool] = lambda: False,   # FIX: abort hook
    trace_callback: Optional[Callable[[str], None]] = None, # The Bridge to Streamlit UI for real-time trace updates
) -> dict:
    """
    Run the nutrition agent and return a structured result dict:
    {
        "response":      str,
        "trace":         list[str],
        "features":      {"agent_used": bool, "rag_used": bool},
        "citations":     list[dict],
        "safety_flags":  list[str],
        "tools_used":    list[str],
        "steps_taken":   int,
        "cached":        bool,
        "aborted":       bool,   ← True when caller requested an early stop
    }

    """
    if chat_history is None:
        chat_history = []
    
    def update_trace_callback(trace):
        if trace_callback:
            trace_callback("\n".join(trace))

    # ── Semantic cache check ──────────────────────────────────────────────
    cache_key = _cache_key(user_message, use_agent, use_rag)
    if cache_key in _cache:
        cached = dict(_cache[cache_key])
        cached["cached"] = True
        cached["trace"]  = ["⚡ Cache hit — serving cached response"] + cached["trace"]
        return cached

    trace    = [f"📥 User Query: {user_message}"]
    update_trace_callback(trace)
    features = {"agent_used": False, "rag_used": False}
    tools_used: list[str]  = []
    citations:  list[dict] = []
    aborted = False

    # ── Safety pre-screening ──────────────────────────────────────────────
    safety_result = screen_message(user_message)
    safety_flags  = safety_result.flags

    if safety_result.mandatory_message:
        trace.append(f"🛡️  Safety gate triggered: {', '.join(safety_flags)}")
        update_trace_callback(trace)
        return {
            "response":     safety_result.mandatory_message,
            "trace":        trace,
            "features":     features,
            "citations":    [],
            "safety_flags": safety_flags,
            "tools_used":   [],
            "steps_taken":  0,
            "cached":       False,
            "aborted":      False,
        }

    for w in (safety_result.warnings or []):
        trace.append(f"⚠️  Safety warning: {w}")
        update_trace_callback(trace)

    # ── Safety injection for system prompt ───────────────────────────────
    safety_notes = ""
    if safety_flags:
        safety_notes = (
            f"\n\nSAFETY CONTEXT: This query has been flagged for: {', '.join(safety_flags)}. "
            "Apply the appropriate clinical caution and include a strong disclaimer."
        )

    system_with_safety = SYSTEM_PROMPT + safety_notes

    _ollama_opts = {
        "num_gpu":     30,
        "num_ctx":     32768*4,
        "temperature": 0.7,
    }

    # ── MODE: Standard LLM (no RAG, no agent tools) ───────────────────────
    # if not use_agent and not use_rag:
    #     trace.append("🔲 Mode: Standard LLM — A1 Baseline (no tools, no RAG)")
    #     update_trace_callback(trace)
    #     messages = (
    #         [{"role": "system", "content": system_with_safety}]
    #         + chat_history
    #         + [{"role": "user", "content": user_message}]
    #     )
    #     response    = ollama.chat(model=MODEL_NAME, messages=messages, options=_ollama_opts)
    #     result_text = response["message"]["content"]
    #     if safety_result.warnings:
    #         result_text += format_safety_disclaimer(safety_result)
    #     result = _build_result(result_text, trace, features, [], safety_flags, [], 0, False)
    #     _cache[cache_key] = result
    #     return result

    # ── MODE: RAG-Only ────────────────────────────────────────────────────
    # if use_rag and not use_agent:
    #     trace.append("📚 Mode: RAG Only — retrieving context before generation")
    #     update_trace_callback(trace)
    #     context_str, chunks = retrieve_as_string(user_message, top_k=3)
    #     features["rag_used"] = bool(chunks)
    #     tools_used.append("retrieve_rag_context")
    #     citations = chunks

    #     rag_system = (
    #         system_with_safety
    #         + "\n\nKNOWLEDGE BASE CONTEXT (cite sources using their labels):\n"
    #         + context_str
    #     )
    #     trace.append(f"📖 RAG retrieved {len(chunks)} relevant document(s)")
    #     update_trace_callback(trace)

    #     messages = (
    #         [{"role": "system", "content": rag_system}]
    #         + chat_history
    #         + [{"role": "user", "content": user_message}]
    #     )
    #     response    = ollama.chat(model=MODEL_NAME, messages=messages, options=_ollama_opts)
    #     result_text = response["message"]["content"]
    #     if safety_result.warnings:
    #         result_text += format_safety_disclaimer(safety_result)

    #     result = _build_result(result_text, trace, features, citations, safety_flags, tools_used, 0, False)
    #     _cache[cache_key] = result
    #     return result

    # ── MODE: Agent (with optional RAG tool) ──────────────────────────────
    active_tools = list()
    if use_rag and use_agent:
        active_tools.append(RAG_TOOL)
        active_tools.extend(AGENT_TOOLS)
        trace.append("🤖 Mode: Full Agent + RAG")

    elif not use_rag and use_agent:
        active_tools.extend(AGENT_TOOLS)
        trace.append("🤖 Mode: Agent Only (no RAG tool)")

    elif use_rag and not use_agent:
        active_tools.append(RAG_TOOL)
        trace.append("📚 Mode: RAG Only — retrieving context before generation")
    else:
        # raise RuntimeError('Issue in implementation!')
        active_tools = None
        trace.append("🔲 Mode: Standard LLM — A1 Baseline (no tools, no RAG)")
    update_trace_callback(trace)

    messages = (
        [{"role": "system", "content": system_with_safety}]
        + chat_history
        + [{"role": "user", "content": user_message}]
    )

    steps_taken   = 0
    final_message: dict = {}

    for step in range(max_steps):
        print(f"--- Agent Step {step + 1} ---")
        steps_taken += 1
        trace.append(f"\n── Step {step + 1} ─────────────────────────────")
        update_trace_callback(trace)

        response   = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=active_tools,
            options=_ollama_opts,
        )
        msg_obj    = response.get("message", {})
        tool_calls = msg_obj.get("tool_calls") or []

        messages.append(msg_obj)
        final_message = msg_obj

        if not tool_calls:
            trace.append("✅ Agent produced final answer — no further tool calls needed")
            update_trace_callback(trace)
            break

        # features["agent_used"] = True

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]

            if fn_name == "retrieve_rag_context":
                features["rag_used"] = True
            elif fn_name in [i['function']['name'] for i in AGENT_TOOLS]:
                features['agent_used'] = True


            if fn_name not in tools_used:
                tools_used.append(fn_name)

            trace.append(
                f"🛠️  Action: calling '{fn_name}' with "
                f"{json.dumps(fn_args, ensure_ascii=False)}" # [:120]
            )
            update_trace_callback(trace)

            # FIX: use timeout-protected call instead of bare func(**fn_args)
            result_str = _call_tool_with_timeout(fn_name, fn_args)

            trace.append(f"👁️  Observation: {result_str}") # result_str[:400]
            update_trace_callback(trace)
            messages.append({"role": "tool", "content": str(result_str), "name": fn_name})

            if fn_name == "retrieve_rag_context" and "CITATIONS:" in result_str:
                _extract_and_store_citations(result_str, citations)

    # ── Synthesise if the loop ended without a text answer ────────────────
    if final_message.get("tool_calls") or not final_message.get("content"):
        if aborted:
            trace.append("🛑 Generating partial answer from data gathered so far…")
        else:
            trace.append("⚠️ Max steps reached — synthesising available data…")
        update_trace_callback(trace)

        summary_response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={**_ollama_opts, "temperature": 0.3},
        )
        final_message = summary_response.get("message", {})
        messages.append(final_message)

    # FIX: single, authoritative assignment of final_content (removed duplicate)
    final_content = final_message.get("content") or "No response generated."
    if safety_result.warnings:
        final_content += format_safety_disclaimer(safety_result)

    result = _build_result(
        final_content, trace, features, citations, safety_flags, tools_used, steps_taken, aborted
    )
    _cache[cache_key] = result
    return result


# ── Helper ────────────────────────────────────────────────────────────────

def _build_result(
    response:     str,
    trace:        list,
    features:     dict,
    citations:    list,
    safety_flags: list,
    tools_used:   list,
    steps_taken:  int,
    aborted:      bool,
) -> dict:
    return {
        "response":     response,
        "trace":        trace,
        "features":     features,
        "citations":    citations,
        "safety_flags": safety_flags,
        "tools_used":   tools_used,
        "steps_taken":  steps_taken,
        "cached":       False,
        "aborted":      aborted,
    }
