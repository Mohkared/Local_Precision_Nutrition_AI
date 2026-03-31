# 🥦 Precision Nutrition AI — Assignment 2

> Qatar University · CMPE 682/783 · Intelligent Systems  
> Upgrade from A1 (pure prompting) → A2 (real RAG + real ReAct agent)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                   │
│   Chat │ Evaluate │ Profile │ About                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  safety.py         │  ← Pre-screening gate
              │  (runs FIRST)      │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  agent.py          │  ← ReAct loop (qwen3.5:4b)
              │  Semantic cache    │    max 5 steps
              └──┬──────────┬──────┘
                 │          │
    ┌────────────▼─┐   ┌────▼────────────┐
    │  rag_engine  │   │   tools.py      │
    │  ChromaDB    │   │   7 real tools  │
    │  MiniLM-L6   │   └─────────────────┘
    └──────┬───────┘
           │
    ┌──────▼───────────────┐
    │  nutrition_knowledge │
    │  10 clinical docs    │
    │  (USDA/ADA/AHA/NIH…) │
    └──────────────────────┘
```

---

## Setup

### 1 — Python dependencies
```bash
pip install -r requirements.txt
```

### 2 — Ollama + model
```bash
# Install Ollama: https://ollama.com/
ollama pull qwen3.5:4b
ollama serve          # keep running in a separate terminal
```

### 3 — Run
```bash
cd Local_Precision_Nutrition_AI/
streamlit run app_streamlit.py
```

The ChromaDB vector store is built automatically on first launch.

---

## Features

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

### Extra Features (beyond assignment)
- **Safety guardrails** — eating disorder detection → mandatory referral; CKD/pregnancy/paediatric flags
- **User profile** — auto-injected into every prompt (age, weight, goal, conditions)
- **LLM-as-judge evaluation** — 5 criteria, 25 test cases, A1 baseline comparison
- **4 Plotly charts** — before/after bar, category breakdown, radar, tool frequency
- **4-tab Streamlit UI** — Chat / Evaluate / Profile / About

---

## Tools (7)

| Tool | Description |
|------|-------------|
| `calculate_tdee_bmi` | Mifflin-St Jeor BMR → TDEE → BMI category + deficit/surplus |
| `calculate_macro_targets` | Macro split by goal (weight_loss / maintenance / muscle_gain / athletic) |
| `get_food_macros` | 35+ USDA-calibrated foods, fuzzy match, per-serving scaling |
| `retrieve_rag_context` | Semantic search over 10 clinical documents |
| `check_supplement_safety` | NIH UL database (15 supplements, risk assessment) |
| `analyze_meal_nutrition` | Parse meal text → sum macros from food DB |
| `calculate_hydration_needs` | WHO-based water target + activity + climate adjustments |


---

## Evaluation

Run from the **Evaluate** tab in the app, or:

```python
# Standalone
from evaluation import run_evaluation_suite
results = run_evaluation_suite(n_cases=25)
```

Metrics: `accuracy`, `actionability`, `safety`, `groundedness` (new), `task_completion` (new) , `citation accuracy` (new), `retrieval relevance` (new) 
LLM judge: `qwen3.5:4b` scores each response 1–5 per criterion.

---

## Project Structure

```
nutrition_ai/
├── app_streamlit.py       # Streamlit UI (4 tabs)
├── agent.py               # ReAct loop
├── tools.py               # 7 tools
├── rag_engine.py          # ChromaDB RAG
├── nutrition_knowledge.py # 10 clinical docs
├── safety.py              # Pre-screening guardrails
├── evaluation.py          # 25-case eval framework
├── requirements.txt
├── README.md
├── chat_history.json      # auto-created when starting a chat
├── eval_results.json      # auto-created when evaluation finishes
├── user_profile.json      # auto-created on first run
├── Sources/               # Sources directory (for RAG)
└── chroma_db/             # auto-created on first run
```

---

## Troubleshooting

**`ModuleNotFoundError: chromadb`**  
→ `pip install chromadb`

**`ollama.ResponseError: model not found`**  
→ `ollama pull qwen3.5:4b`

**Ollama connection refused**  
→ Make sure `ollama serve` is running in another terminal

**`sentence_transformers` slow first load**  
→ Normal — model downloads (~80 MB) on first run, then cached

---

## A1 → A2 Improvements Summary

| Criterion | A1 Score | A2 Target |
|-----------|----------|-----------|
| Nutritional Accuracy | 4.67 / 5.00 | 4.88 / 5.00 |
| Practical Actionability | 4.07 / 5.00 | 4.64 / 5.00 |
| Safety & Boundaries | 5.00 / 5.00 | 5.00 / 5.00 |
| Groundedness | — | 3.28 / 5.00 |
| Task Completion | 4.53 / 5.00 | 4.96 / 5.00 |
| Citation Accuracy | — | 2.84 / 5.00 |
| Retrieval Relevance | — | 3.64 / 5.00 |