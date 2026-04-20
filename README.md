# ⚛️ Study Buddy Physics — Agentic AI Capstone Project

> **Domain:** B.Tech Physics Tutoring  
> **Users:** First/second-year engineering students  
> **LLM:** Anthropic Claude (via LangChain)  
> **Orchestration:** LangGraph with MemorySaver  
> **Knowledge Base:** ChromaDB + SentenceTransformers  
> **UI:** Streamlit  

---

## Problem Statement

B.Tech students need concept clarification outside class hours across **Mechanics, Electromagnetism, Thermodynamics, Optics,** and **Quantum Basics**. They require an AI assistant that:
- Provides grounded explanations from the syllabus (never hallucinates).
- Remembers conversational context across turns.
- Never fabricates formulas or scientific constants.

---

## Architecture

```
User Question
  └─► memory_node          (append to history, extract name, sliding window)
        └─► router_node    (LLM routes: retrieve | tool | skip)
              ├─► retrieval_node    (ChromaDB top-3 semantic search)
              ├─► tool_node         (calculator or datetime)
              └─► skip_retrieval_node (conversational / follow-up)
                    └─► answer_node (grounded generation, never hallucinate)
                          └─► eval_node (faithfulness 0.0–1.0, MAX_RETRIES=2)
                                ├─► [retry] ──► answer_node
                                └─► save_node ──► END
```

**8 LangGraph nodes** | **MemorySaver** with thread_id | **Self-reflection eval loop**

---

## Knowledge Base (10 Documents)

| # | Topic | Coverage |
|---|-------|----------|
| 1 | Newton's Laws | F=ma, inertia, action-reaction |
| 2 | Work Energy Power | W=Fd cosθ, KE=½mv², PE=mgh, W-E theorem |
| 3 | Gravitation | F=Gm₁m₂/r², escape velocity, Kepler's laws |
| 4 | Oscillations | SHM, T=2π√(m/k), pendulum, resonance |
| 5 | Waves | v=fλ, superposition, interference, standing waves |
| 6 | Thermodynamics | All 4 laws, ΔU=Q−W, entropy, ideal gas PV=nRT |
| 7 | Electrostatics | Coulomb's law, Gauss's law, capacitance |
| 8 | Magnetism | F=qv×B, Faraday's law, magnetic flux Φ=BA cosθ |
| 9 | Optics | Snell's law, mirrors, lenses, Young's double-slit |
| 10 | Quantum Basics | E=hf, h=6.626×10⁻³⁴ J·s, de Broglie, Bohr model |

- **Embedder:** `all-MiniLM-L6-v2` (SentenceTransformers)
- **Vector DB:** ChromaDB in-memory (cosine similarity)
- **Top-k:** 3 chunks retrieved per query

---

## Tools

| Tool | Route | Example |
|------|-------|---------|
| `calculator` | `tool` | `0.5 * 2 * 5**2` → `25.0 J` |
| `datetime` | `tool` | Returns current date and time |

Tools **never raise exceptions** — they always return strings.

---

## Project Structure

```
study_buddy_physics/
├── state.py                # CapstoneState TypedDict (10 fields)
├── tools.py                # calculator(), get_datetime()
├── nodes.py                # All 8 node functions + KB + ChromaDB + embedder
├── graph.py                # LangGraph assembly + MemorySaver compilation
├── agent.py                # ask() helper + all mandatory tests + RAGAS
├── capstone_streamlit.py   # Streamlit UI (@st.cache_resource, session_state)
├── tests/
│   ├── __init__.py
│   └── test_questions.py   # pytest: nodes, 10Qs, memory, red-team, tools
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Prerequisites

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Environment Variable

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Linux/macOS
set ANTHROPIC_API_KEY=sk-ant-...        # Windows CMD
$env:ANTHROPIC_API_KEY="sk-ant-..."     # Windows PowerShell
```

### 3. Run Streamlit UI

```bash
streamlit run capstone_streamlit.py
```

Open your browser at `http://localhost:8501`

### 4. Run Full Validation (all tests + RAGAS)

```bash
python agent.py
```

### 5. Run Pytest Test Suite

```bash
pytest tests/test_questions.py -v
```

---

## Capabilities Implemented

| Capability | Status | Detail |
|-----------|--------|--------|
| 8 LangGraph nodes | ✅ | memory, router, retrieve, skip, tool, answer, eval, save |
| 10 ChromaDB documents | ✅ | 100–500 words each, one topic per doc |
| SentenceTransformer embeddings | ✅ | all-MiniLM-L6-v2 |
| MemorySaver + thread_id | ✅ | Sliding window messages[-6:], name extraction |
| Self-reflection eval loop | ✅ | Faithfulness 0.0–1.0, MAX_EVAL_RETRIES=2 |
| Calculator tool | ✅ | Safe eval, never raises |
| Datetime tool | ✅ | Returns current date/time string |
| Router (retrieve/tool/skip) | ✅ | LLM-based, single-word output |
| 10 mandatory test questions | ✅ | Including KE calculation via tool |
| Memory test (Rahul) | ✅ | 3-turn name recall |
| 2 Red-team tests | ✅ | Out-of-scope + prompt injection |
| RAGAS evaluation | ✅ | 5 QA pairs, manual fallback |
| Streamlit UI | ✅ | @st.cache_resource, sidebar, chat |
| Safety rules | ✅ | No hallucination, prompt injection resistance |

---

## Safety Guarantees

- **No hallucination:** Answer node system prompt forbids fabricating formulas.  
- **Graceful unknown:** Responds with *"I do not know based on the available knowledge base."*  
- **Prompt injection resistance:** Sanitised input; refusal logic in answer_node.  
- **Distress handling:** Empathetically redirects to academic advisor / professor.  
- **Faithfulness gating:** Self-reflection loop retries up to 2× if score < 0.7.

---

## Success Criteria

| Criterion | Target | Notes |
|-----------|--------|-------|
| Faithfulness score | ≥ 0.70 | From eval_node + RAGAS |
| 10 test questions | 10/10 PASS | All topics covered |
| Memory persistence | PASS | Rahul name recall across 3 turns |
| Red-team tests | 2/2 PASS | OOS + injection |
| RAGAS baseline | Produced | 5 QA pairs |
| Streamlit UI | Deployed | `streamlit run capstone_streamlit.py` |

---

## Technical Improvement (with more time)

**HyDE + Cross-Encoder Re-Ranking:** Instead of embedding the raw student question for retrieval, first prompt the LLM to generate a *hypothetical ideal answer*, then embed that for vector search (Hypothetical Document Embeddings). Follow this with cross-encoder re-ranking on the top-10 candidates before selecting the final top-3. This pipeline significantly improves retrieval precision for short or vague student questions (e.g., "explain entropy") and would push faithfulness scores consistently above **0.90**.

---

## Environment Notes

- **Windows users:** All `open()` calls use `encoding='utf-8'` as required.
- **ANTHROPIC_API_KEY** must be set before running any module.
- ChromaDB uses in-memory (ephemeral) mode — no disk persistence.
- Re-running resets the KB (module-level initialisation).

---

*Built with: LangGraph · ChromaDB · SentenceTransformers · LangChain-Anthropic · Streamlit · RAGAS · pytest*
