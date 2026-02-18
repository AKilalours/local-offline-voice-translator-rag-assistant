![Cover](local_rag_coverimage.png)
# local-offline-voice-translator-rag-assistant (Whisper + Ollama + Coqui TTS)

A local-first **speech-to-speech translator** and **document-grounded RAG assistant** that runs on your machine using open-source components.  
It supports:

- **Chat Mode (RAG)**: answers questions from local docs with **chunk-level citations** and deterministic refusals when unsupported.
- **Translation Mode**: translates whatever you say (except control commands) into a chosen target language.
- **API Mode (FastAPI)**: exposes `/ask` (RAG) + `/translate` endpoints for integration. 

This project is built to demonstrate **ML/LLM system engineering**: retrieval evaluation, latency benchmarks, security/refusal harness, and **RBAC enforcement**.

**Demo link** : https://drive.google.com/drive/folders/1NNxKn7dPizFucjTRPZCNglUZFNbh4CPI?usp=sharing

This drive link has the Demo video link of how it works
---

## Why this project is not a toy

This is not a “hello-world voice bot.” It includes production-style safeguards and measurable quality gates:

- **Retrieval evaluation harness** (recall@k on answerable questions)
- **Latency benchmarks** (median and p95 for retrieval, rerank, LLM)
- **Security harness** (prompt-injection + exfil attempts must refuse)
- **RBAC enforcement at retrieval time** (public users cannot retrieve confidential chunks)
- **API packaging + CI** (FastAPI + pytest + GitHub Actions)


---

## Architecture (high level)

### Audio pipeline
Microphone → VAD (RMS threshold) → ASR (Faster-Whisper) → decision router

### Chat Mode (RAG)
User query → Retrieval (embeddings + FAISS) → optional rerank (Cross-Encoder) →  
Context prompt (chunk headers + text) → Local LLM (Ollama) → **citation integrity check** → output

### Translation Mode
User speech → translation prompt (Ollama) → post-processing → TTS output


---

## Features

### 1) Speech-to-text (ASR)
- Faster-Whisper transcription (offline)

### 2) Document-grounded Q/A (RAG)
- Retrieval backends:
  - TF-IDF + FAISS (baseline)
  - Dense embeddings + FAISS
  - Dense + Cross-Encoder rerank (default)
- **Coverage gate**: refuse if retrieved context does not cover key query intent/terms.
- **Citation requirement**: Answers must include **valid citations** like `chunk_id=<id>` **or refuse**.
- **Citation integrity:** the assistant cannot cite chunk IDs that were not retrieved.
  
### 3) Translation mode (speech-to-speech)
- Voice commands for mode switching and target language selection
- Everything spoken in translation mode is translated **except** control commands

### 4) Offline constraints
- Offline constraint:** no live web access — news/weather/market queries return a clear offline limitation.


### 5) Phase 5 RBAC enforcement (retrieval-time access control)
- Chunks are labeled (e.g., `public`, `confidential`)
- The user role determines which chunks are eligible for retrieval
- Demonstrated via an automated RBAC evaluation script

### 6) FastAPI integration
- `/ask`: grounded answers with citations
- `/translate`: translation endpoint

### 7) CI
- `pytest` suite runs via GitHub Actions

---

## Metrics (measured)

### Retrieval quality (Phase 3: answerable questions)
- **TF-IDF recall@k**: `1.000 (13/13)`
- **Dense recall@k**: `1.000 (13/13)`
- **Dense + Rerank recall@k**: `1.000 (13/13)`

### Latency (Phase 3 Benchmarks: Dense + Rerank)
- **Dense retrieval**: median ≈ `8–10 ms`, p95 ≈ `9–11 ms`
- **Dense retrieval + rerank**: median ≈ `47–48 ms`, p95 ≈ `48–50 ms`
- **RAG LLM (Ollama)**: median ≈ `4.3 s`, p95 ≈ `4.5 s` (hardware/model dependent)

### Security (Phase 4 security harness)
- **Refusal rate**: `1.000 (6/6)`  
  Includes: “reveal system prompt”, “print full documents”, “ignore rules”, etc.

### RBAC (Phase 5)
- **Pass rate**: `1.000 (4/4)`  
  Public role is blocked from confidential secret retrieval; admin role allowed.

> Note: Latency and LLM timings vary by model and machine; retrieval and rerank are consistently low-latency CPU operations.

---

## Setup

### Prerequisites
- Python 3.12+
- Ollama installed and running
- A local model pulled (example: `mistral`)
- (Optional) System deps for audio/TTS depending on OS

---

# Quick Start

# 1. Create env + install

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

---
# 2. Start Ollama + pull a model

ollama serve
ollama pull mistral
---

# 3. Build indexes

TF-IDF baseline index

- python ingest/build_index.py

Dense index (FAISS + embessings)

- python ingest/build_dense_index.py

---

# 4. Run: Voice APP (interactive)

- python main.py

Voice commands:

- Start transalation mode: start transalation mode
- Set language: translate into French / translate into Spanish
- Stop translation mode: stop translation mode

---

# Run: API (FastAPI)

- python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Open :

Swagger UI : http://127.0.0.1:8000/docs


Example:/ ask

- curl -s http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"text":"What is RAG?","role":"public"}'

Example:/ translate
- curl -s http://127.0.0.1:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Good morning, have a good day.","target_lang":"French"}'


RBAC via API

Pass role (e.g., public vs admin) to enforce which labeled chunks can be retrieved.
---

# Evaluation & Benchmarks

Phase 3 retrieval comparison
- python eval/eval_phase3.py

Phase 4 security harness
- python eval/security_phase4.py

Phase 5 RBAC eval
- python eval/eval_phase5_rbac.py

Unit tests
- pytest -q

---

# What this demonstrates 

•	End-to-end voice + LLM system integration (ASR, translation, TTS)
•	RAG engineering discipline: retrieval evaluation, latency benchmarking
•	Security-minded design: deterministic refusal + injection/exfil resilience
•	Real access control: RBAC at retrieval time
•	Deployable packaging: FastAPI endpoints + CI-ready tests

---
 
Limitations / Next improvements
•	Add streaming ASR/LLM response for lower perceived latency
•	Replace RMS VAD with a more robust VAD (e.g., WebRTC VAD)
•	Add structured telemetry (JSON logs), traces, and request IDs
•	Add “citation faithfulness” scoring and regression gating in CI
•	Working on more than one line 
---
