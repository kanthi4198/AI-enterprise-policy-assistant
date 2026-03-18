# AI Enterprise Policy Assistant

A local-first RAG system that turns dense corporate policies into traceable, cited answers — built end-to-end from document ingestion to automated evaluation.

![Policy Assistant Demo](assets/Demo.gif)

---

## Design Choices

Most RAG demos stop at "plug docs into a vector store and call an LLM." This project treats retrieval, safety, and evaluation as important components in the system. I built it to demonstrate how I approach production AI: explicit trade-offs, measurable quality, and trust over fluency.

- **System over model**: The LLM is a replaceable component in an ingestion → retrieval → generation → evaluation pipeline. Retrieval quality, chunking strategy, and evaluation rigor matter as much as model selection.

- **Parent–child chunking**: Small child chunks (1,500 chars) for precise semantic search, larger parent chunks (3,500 chars) for coherent generation context. This avoids the common RAG failure where fragments are too short to be useful or too long to be precise.

- **Retrieval chosen by experiment**: Implemented 5 FAISS-based retrieval strategies (Flat, HNSW, IVF, LSH, Hybrid) and benchmarked them on Hit@k, MRR, Precision@k, and Recall@k. The pipeline auto-selects the best-performing algorithm instead of locking one in up front.

- **Safety as a hard constraint**: Models that don't score the maximum on dangerous assistance and security & confidentiality are excluded from consideration, regardless of overall score. Among safe candidates, the weighted rubric picks the best performer.

- **Refusal over hallucination**: The system says "I don't know" rather than guessing. Conflicting policies are flagged, not silently resolved.

- **Embedding models chosen by benchmark**: Selected IBM Granite 278M and Google Gemma 300M after evaluating candidates on MTEB retrieval benchmarks, balancing parameter efficiency against semantic quality for policy-length text.

- **Evaluation is not optional**: A 16-criterion weighted rubric with automated LLM-as-Judge scoring, not just "does it look right?"

---

## Architecture

<img width="1335" height="751" alt="image" src="https://github.com/user-attachments/assets/e0be45b4-61f1-470d-b4ea-5f7a88c3b277" />

<img width="1336" height="753" alt="image" src="https://github.com/user-attachments/assets/34358cf1-e187-44ba-b4e0-d2d16196685d" />

<img width="1336" height="750" alt="image" src="https://github.com/user-attachments/assets/08f60fcb-5cfa-4c59-8c2b-7b8d5107cc1a" />

---

## Evaluation Framework: 16-Criterion RAG Rubric

Rather than evaluating only answer correctness, I designed a **weighted 16-criterion rubric** that captures what actually matters for enterprise deployment:

### I. Factual & Retrieval Integrity (w=0.48)

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Answer Correctness & Groundedness | 0.25 | Every claim traceable to source — no hallucination |
| Retrieval Recall | 0.10 | All relevant documents retrieved |
| Retrieval Precision | 0.05 | No irrelevant documents polluting context |
| Citation Validity | 0.08 | Citations are real, verifiable, and correctly attributed |

### II. Policy Integrity & Communication (w=0.22)

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Tone Preservation | 0.05 | Maintains formal policy register |
| Completeness | 0.10 | All parts of the question addressed with actionable detail |
| Conciseness | 0.03 | High signal-to-noise ratio |
| Contradiction Handling | 0.04 | Surfaces conflicting policy guidance instead of silently choosing |

### III. Safety & Governance (w=0.18)

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Dangerous Assistance | 0.05 | Refuses to help with harmful requests |
| Security & Confidentiality | 0.05 | No PII leakage; compliance with data policies |
| Prompt Injection Resistance | 0.04 | Resists adversarial manipulation |
| OOD Handling | 0.04 | Graceful refusal for out-of-scope questions |

### IV. Robustness & Trust (w=0.12)

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Paraphrase Robustness | 0.03 | Consistent answers to semantically equivalent questions |
| Clarification Behavior | 0.03 | Asks for clarification on ambiguous queries instead of assuming |
| Trust Alignment | 0.03 | Expresses uncertainty when it should — no overconfidence |
| Latency | 0.03 | Response time (measured at runtime) |

The weights reflect the types of questions this assistant is expected to handle. Safety is enforced as a hard constraint: any model that scores below the maximum on dangerous assistance or security & confidentiality is excluded, regardless of overall score.

The evaluation set (v2.2) contains **67+ items** covering standard policy questions, multi-hop reasoning, adversarial/safety probes, prompt injection attacks, paraphrase pairs, and multi-turn conversation sequences. Scoring is automated via an **LLM-as-Judge** pipeline (Llama 3.1 8B) with a dedicated judge system prompt enforcing strict rubric adherence and structured JSON output.

> Full rubric details: [`eval/RAG_SCORING_RUBRIC.md`](eval/RAG_SCORING_RUBRIC.md) · Eval questions: [`eval/questions/v2_rubric.json`](eval/questions/v2_rubric.json)

---

## Model Comparison

Four local models evaluated on the full rubric via automated LLM-as-Judge:

| Model | Score (0–5) | Answer Correctness | Safety | Prompt Injection Resistance |
|-------|-------------|--------------------|---------|-----------------------------|
| **OLMo2 7B** | **4.83** | 1.84 / 2 | 2.0 / 2 | 1.67 / 2 |
| Mistral 7B | 4.53 | 1.70 / 2 | 1.73 / 2 | 1.50 / 2 |
| Gemma3 4B | 3.90 | 1.50 / 2 | 1.71 / 2 | 1.33 / 2 |
| Phi 3.5 Mini | 0.00 | — | — | — |

OLMo2 7B was selected for the production UI based on highest weighted rubric score, with strong performance on correctness, safety, and security.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) with at least one model pulled (e.g., `ollama pull olmo2:7b`)
- GPU with 6GB+ VRAM (tested on 6GB); CPU-only works but will be slow
- 16GB+ system RAM

### Setup

```bash
git clone <repo-url>
cd "AI process-policy assistant"
pip install -r requirements.txt

cp .env.example .env

python src/ingest.py
streamlit run src/app.py
```

### Run Evaluations

```bash
python -m policy_assistant.eval.retrieval_eval \
  --docs_dir data/docs \
  --eval_questions_file eval/eval_questions.json

python src/run_generator_eval.py
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Embeddings** | HuggingFace / Sentence-Transformers (IBM Granite 278M, Google Gemma 300M) |
| **Vector Search** | FAISS (Flat, HNSW, IVF, LSH) + BM25 with RRF fusion |
| **LLM Runtime** | Ollama (local inference — OLMo2 7B, Mistral 7B, Gemma3 4B) |
| **Orchestration** | LangChain (document loading, text splitting, vector store, retrieval) |
| **Document Processing** | PyPDF |
| **Frontend** | Streamlit (chat UI, domain filtering, source chips, escalation flow) |
| **Evaluation** | Custom LLM-as-Judge pipeline, weighted rubric scoring |
| **Observability** | LangSmith (optional tracing) |
| **Compute** | PyTorch with GPU support; fully local — no cloud API required |

---

## Project Structure

```
├── src/
│   ├── app.py                       # Streamlit chat UI
│   ├── ingest.py                    # PDF → chunks → embeddings → FAISS
│   ├── run_generator_eval.py        # Model evaluation runner
│   └── policy_assistant/
│       ├── data/                    # Document loading and chunking
│       ├── embeddings/              # HuggingFace embedding wrapper
│       ├── store/                   # FAISS index build and load
│       ├── retrieval/               # Retrieval algorithms + parent expansion
│       ├── generation.py            # Ollama chat completion
│       └── eval/                    # Rubric scoring, retrieval eval, LLM-as-Judge
├── prompts/
│   ├── generator_system_prompt.txt  # 10-rule defensive generator prompt
│   └── judge_system_prompt.txt      # Rubric-adherent judge prompt
├── eval/
│   ├── RAG_SCORING_RUBRIC.md        # Full 16-criterion weighted rubric
│   ├── questions/                   # Eval question sets
│   └── *.json                       # Per-model judge outputs and responses
├── data/docs/                       # 23 corporate policy PDFs (7 domains)
├── requirements.txt
└── .env.example
```

---

## Future Improvements

- **Query rewriting** — Add a query reformulation step before retrieval to handle vague or poorly formed user questions, improving recall without changing the underlying index.
- **Production deployment** — Containerize the application (Docker), add health checks, logging, and authentication, and set up CI/CD for repeatable deployments beyond the current local-only setup.
