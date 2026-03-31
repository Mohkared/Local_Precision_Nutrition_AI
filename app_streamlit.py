"""
app.py
────────────────────────────────────────────────────────────────────────────
Precision Nutrition AI — Enhanced A2 Streamlit Application
Tabs:
  1. 💬 Chat       — main chat with RAG/Agent toggles + citations
  2. 📊 Evaluate   — run 25-case evaluation, view Plotly charts
  3. 👤 Profile    — save user stats that auto-fill agent context
  4. ℹ️  About      — architecture diagram + system info

To run:
    streamlit run app.py
────────────────────────────────────────────────────────────────────────────
"""
import subprocess
import json
import time
import os
import uuid
from datetime import datetime

import streamlit as st

from agent import run_nutrition_agent
from rag_engine import init_rag, get_kb_stats, add_custom_document
from evaluation import (
    TEST_CASES,
    evaluate_test_case,
    build_summary_dataframe,
    make_before_after_chart,
    make_category_breakdown_chart,
    make_radar_chart,
    make_tool_usage_chart,
    make_rag_metrics_chart,   # new chart for RAG-specific metrics
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Precision Nutrition AI",
    page_icon="🥗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Persistence files ──────────────────────────────────────────────────────
CHAT_FILE    = "chat_history.json"
PROFILE_FILE = "user_profile.json"
EVAL_FILE    = "eval_results.json"

# # ══════════════════════════════════════════════════════════════════════════
# # Enable Ollama
# # ══════════════════════════════════════════════════════════════════════════
# # Trigger the auto-start
# subprocess.run(["ollama", "list"], capture_output=True)
# time.sleep(2) # Give the engine a heartbeat to stabilize


# ══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════

def _load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# RAG initialisation (once per session)
# ══════════════════════════════════════════════════════════════════════════

if "rag_ready" not in st.session_state:
    with st.spinner("🔧 Initialising RAG knowledge base (first run may take ~30 s)…"):
        stats = init_rag()
    st.session_state.rag_ready = True
    st.session_state.rag_stats = stats


# ══════════════════════════════════════════════════════════════════════════
# Session state defaults
# ══════════════════════════════════════════════════════════════════════════

if "chats" not in st.session_state:
    st.session_state.chats = _load_json(CHAT_FILE, {})

if not st.session_state.chats:
    init_id = str(uuid.uuid4())
    st.session_state.chats[init_id] = {
        "title": "New Chat", "messages": [], "updated_at": datetime.now().isoformat()
    }
    st.session_state.current_chat_id = init_id
    _save_json(CHAT_FILE, st.session_state.chats)

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]

if "user_profile" not in st.session_state:
    st.session_state.user_profile = _load_json(PROFILE_FILE, {})

if "eval_results" not in st.session_state:
    st.session_state.eval_results = _load_json(EVAL_FILE, [])

if "food_diary" not in st.session_state:
    st.session_state.food_diary = []

# agent_running flag
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/salad.png", width=60)
    st.title("Precision Nutrition AI")
    st.caption("Assignment 2 — RAG + Agent Enhanced")

    # ── Feature toggles ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ System Mode")

    mode = st.selectbox(
        "Active Mode",
        [
            "🚀 Full System (RAG + Agent)",
            "📚 RAG Only",
            "🤖 Agent Only",
            "🔲 A1 Baseline (Prompting Only)",
        ],
        help=(
            "Choose which capabilities are active:\n"
            "• Full System = RAG retrieval + all 7 tools\n"
            "• RAG Only = inject context but no tool loop\n"
            "• Agent Only = tool loop without knowledge base\n"
            "• A1 Baseline = plain LLM, no enhancements"
        ),
        key="active_mode_selection"  # <--- Add this!
    )

    use_rag   = "RAG"   in mode or "Full" in mode
    use_agent = "Agent" in mode or "Full" in mode

    rag_badge   = "🟢 RAG"   if use_rag   else "🔴 RAG"
    agent_badge = "🟢 Agent" if use_agent else "🔴 Agent"
    st.caption(f"{rag_badge} &nbsp; {agent_badge}")

    # ── Knowledge base status ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    kb_stats = st.session_state.get("rag_stats", {})
    if "error" in kb_stats:
        st.error(f"RAG error: {kb_stats['error']}")
    else:
        st.metric("Documents indexed", kb_stats.get("total_documents", "—"))
        st.caption(f"Model: {kb_stats.get('embedding_model', '—')}")
        st.caption(f"Similarity threshold: dist ≤ {kb_stats.get('similarity_thresh', '—')}")

    with st.expander("➕ Add custom document"):
        custom_text   = st.text_area("Paste document text", height=100)
        custom_source = st.text_input("Source name", placeholder="e.g. My clinic guidelines")
        if st.button("Add to knowledge base") and custom_text and custom_source:
            doc_id = add_custom_document(custom_text, custom_source)
            st.success(f"Added! Document ID: {doc_id}")
            st.session_state.rag_stats = get_kb_stats()
            st.rerun()

    # ── Chat history ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Conversations")
    if st.button("➕ New Chat", width="stretch"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "title": "New Chat", "messages": [], "updated_at": datetime.now().isoformat()
        }
        st.session_state.current_chat_id = new_id
        _save_json(CHAT_FILE, st.session_state.chats)
        st.rerun()

    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1].get("updated_at", ""),
        reverse=True,
    )
    for cid, cdata in sorted_chats:
        is_active = cid == st.session_state.current_chat_id
        label     = f"{'🟢' if is_active else '📄'} {cdata['title']}"
        if st.button(label, key=f"chat_{cid}", width="stretch"):
            st.session_state.current_chat_id = cid
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════

tab_chat, tab_eval, tab_profile, tab_about = st.tabs(
    ["💬 Chat", "📊 Evaluate", "👤 Profile", "ℹ️ About"]
)


# ─────────────────────────────────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.header("🥗 Precision Nutrition AI")
    st.caption("Ask any nutrition question. Toggle the system mode in the sidebar to compare capabilities.")

    active_chat = st.session_state.chats[st.session_state.current_chat_id]

    # 1. Create a scrollable container for the conversation
    # This ensures content stays ABOVE the fixed chat input bar.
    chat_container = st.container()

    # 2. Render existing messages inside the container
    with chat_container:
        for msg in active_chat["messages"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    f       = msg.get("features", {})
                    b_rag   = "🟢" if f.get("rag_used")   else "🔴"
                    b_agent = "🟢" if f.get("agent_used") else "🔴"
                    cached  = "⚡ cached" if msg.get("cached") else ""
                    aborted = "🛑 aborted" if msg.get("aborted") else ""
                    badges  = f"{b_rag} RAG &nbsp;|&nbsp; {b_agent} Agent"
                    if cached: badges += f" &nbsp;|&nbsp; {cached}"
                    if aborted: badges += f" &nbsp;|&nbsp; {aborted}"
                    st.caption(badges)

                st.markdown(msg["content"])

                if msg.get("safety_flags"):
                    st.warning(f"⚠️ Safety flags: {', '.join(msg['safety_flags'])}")

                if msg.get("citations"):
                    with st.expander("📚 Sources & Citations"):
                        for c in msg["citations"]:
                            st.markdown(f"**{c.get('label', '?')}** — {c.get('text', '?')[len(c.get('label', '')):]}")

                if msg.get("trace"):
                    with st.expander("🔍 Reasoning Trace"):
                        for line in msg["trace"]:
                            st.text(line)

        # 3. Handle Agent Execution Logic if running
        if st.session_state.get("agent_running") and active_chat["messages"] and active_chat["messages"][-1]["role"] == "user":
            with st.chat_message("assistant"):
                # Use st.status to keep the "Thinking" bubble inside the chat flow
                with st.status("Thinking… (use 🛑 Stop at the top to interrupt)", state="running") as status:
                    # 1. Create a "Live Window" for the logs
                    log_window = st.empty()

                    # 2. Define the callback that 'agent.py' will call
                    def my_ui_bridge(new_trace_text):
                        # This updates the window without needing write_stream or a loop
                        log_window.markdown(f"```text\n{new_trace_text}\n```")
                    



                    # status.write_stream()
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in active_chat["messages"][:-1]
                    ]
                    
                    # Profile context injection
                    profile = st.session_state.user_profile
                    last_query = active_chat["messages"][-1]["content"]
                    if profile:
                        # profile_ctx = f"[USER PROFILE: Gender: {profile.get('gender','?')}, Age: {profile.get('age','?')}, Weight: {profile.get('weight_kg','?')}kg, Height: {profile.get('height_cm','?')}cm, Activity Level: {profile.get('activity','?')}, Goal: {profile.get('goal','?')}]\n\n"
                        profile_ctx = (
                            f"[USER PROFILE — use this if relevant: "
                            f"Age {profile.get('age', '?')}, "
                            f"Gender {profile.get('gender', '?')}, "
                            f"Weight {profile.get('weight_kg', '?')} kg, "
                            f"Height {profile.get('height_cm', '?')} cm, "
                            f"Activity: {profile.get('activity', '?')}, "
                            f"Goal: {profile.get('goal', '?')}]\n\n"
                            f"Medical Conditions: {profile.get('conditions', 'None')}\n\n"
                        )
                        last_query = profile_ctx + last_query

                    try:
                        result = run_nutrition_agent(
                            user_message=last_query,
                            chat_history=history,
                            use_agent=use_agent,
                            use_rag=use_rag,
                            # should_stop=_should_stop,
                            trace_callback=my_ui_bridge,  # Pass the callback to the agent
                        )
                    except Exception as e:
                        active_chat["messages"].append({
                            "role": "assistant",
                            "content": f"⚠️ **System Error:** {str(e)}",
                            "is_error": True  # Custom flag to help with formatting
                        })
                        st.session_state.agent_running = False
                        _save_json(CHAT_FILE, st.session_state.chats)
                        st.rerun()
                    # result = run_nutrition_agent(
                    #     user_message=last_query,
                    #     chat_history=history,
                    #     use_agent=use_agent,
                    #     use_rag=use_rag,
                    #     # should_stop=_should_stop,
                    #     trace_callback=my_ui_bridge,  # Pass the callback to the agent
                    # )

                    status.update(label="Response generated!", state="complete", expanded=False)

                # Save to history and reset flags
                active_chat["messages"].append({
                    "role":         "assistant",
                    "content":      result["response"],
                    "trace":        result["trace"],
                    "features":     result["features"],
                    "citations":    result.get("citations",    []),
                    "safety_flags": result.get("safety_flags", []),
                    "cached":       result.get("cached",       False),
                    "aborted":      result.get("aborted",      False),
                })
                active_chat["updated_at"] = datetime.now().isoformat()
                _save_json(CHAT_FILE, st.session_state.chats)
                
                st.session_state.agent_running = False
                st.rerun()

    # 4. Place Chat Input bar at the VERY BOTTOM of the tab
    # By placing it after the container, Streamlit pins it to the bottom.
    user_input = st.chat_input("Ask a nutrition question…", key="bottom_pinned_input", disabled=st.session_state.agent_running)

    if user_input:
        # Reset flags and add user message
        st.session_state.agent_running   = True
        
        if active_chat["title"] == "New Chat":
            active_chat["title"] = " ".join(user_input.split()[:5]) + "…"
            
        active_chat["messages"].append({"role": "user", "content": user_input})
        active_chat["updated_at"] = datetime.now().isoformat()
        _save_json(CHAT_FILE, st.session_state.chats)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# TAB 2: EVALUATE
# ─────────────────────────────────────────────────────────────────────────
with tab_eval:
    st.header("📊 Evaluation Dashboard")
    st.markdown(
        "Run the 25-case test suite with LLM-as-judge scoring across **7 criteria**. "
        "Compares A1 Baseline vs A2 Enhanced system."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        eval_mode = st.selectbox(
            "Evaluate in mode",
            ["Full System (RAG + Agent)", "RAG Only", "A1 Baseline"],
        )
    with col_b:
        n_cases = st.slider("Number of cases to run", 1, 25, 5)
    with col_c:
        subset = st.multiselect(
            "Filter by category",
            ["Typical", "Clinical", "Edge", "Agent-RAG"],
            default=["Typical", "Clinical", "Edge", "Agent-RAG"],
        )

    filtered_cases = [tc for tc in TEST_CASES if tc["category"] in subset][:n_cases]
    st.info(f"Will evaluate {len(filtered_cases)} test case(s).")

    if st.button("▶️ Run Evaluation", type="primary"):
        _use_rag   = "RAG"   in eval_mode or "Full" in eval_mode
        _use_agent = "Agent" in eval_mode or "Full" in eval_mode
        results    = []

        prog = st.progress(0, text="Running evaluation…")
        for i, tc in enumerate(filtered_cases):
            print(f"🟡 Evaluating {tc['id']} — mode: RAG={_use_rag}, Agent={_use_agent}")
            prog.progress(i / len(filtered_cases), text=f"Evaluating {tc['id']} — {tc['label']}")
            try:
                res = evaluate_test_case(tc, run_nutrition_agent, _use_agent, _use_rag)
            except Exception as e:
                print(f'⚠️ **System Error with testcase {tc["id"]}:** {str(e)}')
                continue

            results.append(res)
            existing_ids = {r["id"] for r in st.session_state.eval_results}
            if res["id"] in existing_ids:
                st.session_state.eval_results = [
                    res if r["id"] == res["id"] else r
                    for r in st.session_state.eval_results
                ]
            else:
                st.session_state.eval_results.append(res)

        prog.progress(1.0, text="✅ Evaluation complete!")
        _save_json(EVAL_FILE, st.session_state.eval_results)
        st.success(f"Evaluated {len(results)} cases. Results saved.")

    if st.session_state.eval_results:
        all_res = st.session_state.eval_results
        st.markdown("---")

        # ── Summary metrics ───────────────────────────────────────────
        avg_overall  = sum(r["overall"] for r in all_res) / len(all_res)
        avg_safety   = sum(r["scores"].get("safety",             {}).get("score", 0) for r in all_res) / len(all_res)
        avg_accuracy = sum(r["scores"].get("accuracy",           {}).get("score", 0) for r in all_res) / len(all_res)
        avg_cit_acc  = sum(r["scores"].get("citation_accuracy",  {}).get("score", 0) for r in all_res) / len(all_res)
        avg_ret_rel  = sum(r["scores"].get("retrieval_relevance",{}).get("score", 0) for r in all_res) / len(all_res)
        avg_steps    = sum(r["steps_taken"] for r in all_res) / len(all_res)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Overall (avg)",    f"{avg_overall:.2f}/5")
        m2.metric("Safety (avg)",     f"{avg_safety:.2f}/5")
        m3.metric("Accuracy (avg)",   f"{avg_accuracy:.2f}/5")
        m4.metric("Citation Acc.",    f"{avg_cit_acc:.2f}/5")
        m5.metric("Retrieval Rel.",   f"{avg_ret_rel:.2f}/5")
        m6.metric("Avg Steps",        f"{avg_steps:.1f}")

        # ── Score table ───────────────────────────────────────────────
        st.subheader("Score Table")
        df = build_summary_dataframe(all_res)
        st.dataframe(
            df.style.background_gradient(subset=["Overall"], cmap="RdYlGn", vmin=1, vmax=5),
            width="stretch",
            hide_index=True,
        )

        # ── Charts ────────────────────────────────────────────────────
        st.subheader("Visualisations")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_before_after_chart(all_res),      width="stretch")
        with c2:
            st.plotly_chart(make_radar_chart(all_res),             width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(make_category_breakdown_chart(all_res),width="stretch")
        with c4:
            st.plotly_chart(make_tool_usage_chart(all_res),        width="stretch")

        # FIX: new RAG-specific metrics chart (row 3) ─────────────────
        st.plotly_chart(make_rag_metrics_chart(all_res), width="stretch")

        # ── Per-case details ──────────────────────────────────────────
        st.subheader("Per-Case Details")
        selected_id = st.selectbox(
            "Select test case to inspect",
            [r["id"] for r in all_res],
            format_func=lambda x: f"{x} — {next(r['label'] for r in all_res if r['id']==x)}",
        )
        sel = next(r for r in all_res if r["id"] == selected_id)

        st.markdown(f"**Query:** {sel['query']}")
        st.markdown(f"**Response:**\n{sel['response']}")

        cols = st.columns(len(sel["scores"]))
        for col, (crit, info) in zip(cols, sel["scores"].items()):
            col.metric(crit.replace("_", " ").title(), f"{info['score']}/5", help=info.get("reason", ""))

        tools_str = ", ".join(sel["tools_used"]) or "none"
        st.caption(
            f"Tools: {tools_str} | Steps: {sel['steps_taken']} | "
            f"Latency: {sel['latency_s']} s | Safety flags: {sel['safety_flags'] or 'none'}"
        )

        if sel.get("citations"):
            with st.expander("📚 Sources & Citations"):
                for c in sel["citations"]:
                    st.markdown(f"{c.get('text', '?')}")

        if sel.get("trace"):
            with st.expander("🔍 Reasoning Trace"):
                for line in sel["trace"]:
                    st.text(line)

        if st.button("🗑️ Clear all evaluation results"):
            st.session_state.eval_results = []
            _save_json(EVAL_FILE, [])
            st.rerun()

    else:
        st.info("No evaluation results yet. Click 'Run Evaluation' to start.")


# ─────────────────────────────────────────────────────────────────────────
# TAB 3: USER PROFILE
# ─────────────────────────────────────────────────────────────────────────
with tab_profile:
    st.header("👤 User Profile")
    st.markdown(
        "Save your stats here — they will automatically be injected as context in every "
        "chat message so you don't have to repeat yourself."
    )

    prof = st.session_state.user_profile
    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            age       = st.number_input("Age",         min_value=10,  max_value=100,   value=int(prof.get("age", 25)))
            weight_kg = st.number_input("Weight (kg)", min_value=20.0,max_value=300.0, value=float(prof.get("weight_kg", 70.0)))
            height_cm = st.number_input("Height (cm)", min_value=100.0,max_value=250.0,value=float(prof.get("height_cm", 170.0)))
        with c2:
            gender   = st.selectbox("Gender", ["male", "female"],
                                    index=0 if prof.get("gender", "male") == "male" else 1)
            activity = st.selectbox("Activity Level", [
                "Sedentary (1.2)", "Lightly Active (1.375)",
                "Moderately Active (1.55)", "Very Active (1.725)", "Extra Active (1.9)"
            ], index=["Sedentary (1.2)", "Lightly Active (1.375)", "Moderately Active (1.55)",
                      "Very Active (1.725)", "Extra Active (1.9)"].index(
                          prof.get("activity", "Moderately Active (1.55)")))
            goal = st.selectbox("Goal", [
                "weight_loss", "maintenance", "muscle_gain", "athletic_performance"
            ], index=["weight_loss", "maintenance", "muscle_gain", "athletic_performance"].index(
                prof.get("goal", "maintenance")))
        conditions = st.multiselect(
            "Medical conditions (for context)",
            ["Type 2 Diabetes", "Hypertension", "High Cholesterol",
             "Kidney Disease", "Pregnancy", "Celiac Disease", "Lactose Intolerance"],
            default=prof.get("conditions", []),
        )
        allergies = st.text_input("Allergies / intolerances", value=prof.get("allergies", ""))

        if st.form_submit_button("💾 Save Profile"):
            st.session_state.user_profile = {
                "age": age, "weight_kg": weight_kg, "height_cm": height_cm,
                "gender": gender, "activity": activity, "goal": goal,
                "conditions": conditions, "allergies": allergies,
            }
            _save_json(PROFILE_FILE, st.session_state.user_profile)
            st.success("Profile saved! It will be used to personalise your chat responses.")

    # ── Food diary ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🍽️ Food Diary (Today)")
    st.caption("Track what you've eaten today — running totals update automatically.")

    with st.form("diary_form"):
        d1, d2 = st.columns([3, 1])
        with d1:
            food_name = st.text_input("Food item", placeholder="e.g. chicken breast")
        with d2:
            grams = st.number_input("Grams", min_value=1, max_value=2000, value=100)
        if st.form_submit_button("Add to diary") and food_name:
            from tools import get_food_macros
            macro_str = get_food_macros(food_name, grams)
            st.session_state.food_diary.append({
                "food": food_name, "grams": grams, "info": macro_str,
                "time": datetime.now().strftime("%H:%M"),
            })

    if st.session_state.food_diary:
        import re
        total_cal = total_protein = total_carbs = total_fat = 0.0
        for entry in st.session_state.food_diary:
            st.markdown(f"**{entry['time']}** — {entry['food']} ({entry['grams']}g)")
            st.caption(entry["info"][:200])
            m_cal  = re.search(r"Calories\s*:\s*([\d.]+)", entry["info"])
            m_prot = re.search(r"Protein\s*:\s*([\d.]+)", entry["info"])
            m_carb = re.search(r"Carbs\s*:\s*([\d.]+)", entry["info"])
            m_fat  = re.search(r"Fat\s*:\s*([\d.]+)", entry["info"])
            if m_cal:  total_cal     += float(m_cal.group(1))
            if m_prot: total_protein += float(m_prot.group(1))
            if m_carb: total_carbs   += float(m_carb.group(1))
            if m_fat:  total_fat     += float(m_fat.group(1))

        st.markdown("**Daily Totals:**")
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Calories", f"{total_cal:.0f} kcal")
        dc2.metric("Protein",  f"{total_protein:.0f} g")
        dc3.metric("Carbs",    f"{total_carbs:.0f} g")
        dc4.metric("Fat",      f"{total_fat:.0f} g")

        if st.button("🗑️ Clear diary"):
            st.session_state.food_diary = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# TAB 4: ABOUT
# ─────────────────────────────────────────────────────────────────────────
with tab_about:
    st.header("ℹ️ About This System")

    st.markdown("""
## Assignment 2 — Track A (RAG) + Track B (Agent) Combined

This system enhances the A1 nutrition counselling chatbot with:

| Feature | A1 Baseline | A2 Enhanced |
|---------|------------|-------------|
| RAG | ❌ | ✅ Real ChromaDB + sentence-transformers |
| Tools | ❌ | 7 real tools |
| Food DB | ❌ | 10+ items |
| Safety | Reactive (in prompt) | + Pre-screening guardrail module |
| Citations | ❌ | ✅ Source labels in responses |
| Multi-turn | ❌ | ✅ + user profile injection |
| Evaluation | by non-expert humans | 25-case LLM-as-judge suite (7 criteria) |
| UI | ❌ | 4 tabs (Chat, Evaluate, Profile, About) |
| Caching | ❌ | ✅ Semantic cache |
| Tool timeout | ❌ | ✅ 30 s via ThreadPoolExecutor |
| User abort | ❌ | ✅ Stop button |

---

## Architecture Diagram
```
┌─────────────────────────────────────────────────────┐
│                    app.py (Streamlit)               │
│  [Chat Tab] [Evaluate Tab] [Profile Tab] [About]    │
│                                                     │
└────────────────────┬────────────────────────────────┘
                     │ user query + profile context
                     ▼
┌─────────────────────────────────────────────────────┐
│                   safety.py                         │
│  Pre-screening: eating disorders, paediatric, CKD   │
│  → short-circuit with mandatory message if critical │
└────────────────────┬────────────────────────────────┘
                     │ safe query
                     ▼
┌────────────────────────────────────────────────────┐
│                   agent.py                         │
│  ┌──────────────────────────────────────────────┐  │
│  │ Semantic Cache (in-process, session-scoped)  │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ReAct Loop (checks should_stop() each step):      │
│  Think → [call tool w/ 30s timeout] → Observe      │
│       → repeat → Answer                            │
└──────┬──────────────────────┬──────────────────────┘
       │                      │
       ▼                      ▼
┌────────────┐     ┌─────────────────────────────────┐
│ rag_engine │     │         tools.py                │
│            │     │  1. calculate_tdee_bmi          │
│ ChromaDB   │     │  2. calculate_macro_targets     │
│ MiniLM-L6  │     │  3. get_food_macros (35+ items) │
│ dist ≤0.40 │     │  4. retrieve_rag_context        │
└────────────┘     │  5. check_supplement_safety     │
                   │  6. analyze_meal_nutrition      │
                   │  7. calculate_hydration_needs   │
                   └─────────────────────────────────┘
```
""")

    st.info(
        "**Stack:** Streamlit · Ollama (qwen3.5:4b) · ChromaDB · "
        "sentence-transformers (all-MiniLM-L6-v2 (ONNX)) · Plotly · Pandas"
    )
