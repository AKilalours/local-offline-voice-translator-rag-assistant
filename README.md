# Local Offline Voice Translator RAG Assistant

## Overview

This project implements a fully local **voice assistant and speech‑to‑speech translator** that runs entirely on a Mac, using only open‑source models:

- **Whisper (faster‑whisper)** for speech‑to‑text (ASR)  
- **Ollama (Mistral)** as the local LLM for chat and translation prompts  
- **SentenceTransformers** for a lightweight RAG context  
- **Coqui TTS (VCTK VITS)** for high‑quality English speech synthesis [web:20][web:159][web:50]

The assistant:

- Listens to microphone audio  
- Transcribes with Whisper  
- Routes the text into **chat mode** or **translation mode** based on **voice commands**  
- Uses a local LLM via Ollama either to answer questions or generate translations  
- Speaks the response using Coqui TTS

Example commands:

- “Translate into Spanish.”  
- “Translate into French.”  
- “Translate into German.”  
- “Stop translation mode.”  
- “What language are you translating into?”

---

## Architecture

### Components

- **ASR:** `faster-whisper` (“small” model) running on CPU for real‑time transcription of microphone input. [web:159]  
- **Embedding / RAG:** SentenceTransformers (`all‑MiniLM‑L6‑v2`) + cosine similarity over a small corpus describing the system, used to provide context for the LLM.  
- **LLM:** `mistral` served locally via **Ollama**’s `/api/generate` endpoint for chat and translation prompts. [web:227]  
- **TTS:** Coqui `tts_models/en/vctk/vits` for English multi‑speaker synthesis. [web:50][web:131]  
- **Audio I/O:** `sounddevice` + `soundfile` for streaming microphone input and playing WAV output.

### Modes

- **Chat mode**

  ```text
  Mic → Whisper ASR → RAG (SentenceTransformers + cosine) → Ollama “assistant” prompt → Coqui TTS

 - **Transalation mode**
   Mic → Whisper ASR → Ollama “translate into <lang>” prompt → Coqui TTS

### Control Flow
Mic
  ↓
Whisper ASR (text)
  ↓
If text is a command:
    - "Translate into <lang>"  → set mode=translate, current_target_lang=<lang>
    - "Start translation mode" → mode=translate
    - "Stop translation mode" / "Stop translation" → mode=chat
    - "What language are you translating into?" → spoken status
Else:
    If mode=chat:
        → RAG + LLM answer → TTS
    If mode=translate:
        → LLM translation (current_target_lang) → TTS

### Features

* Fully local, offline‑capable pipeline using Whisper (ASR), a local LLM via Ollama, SentenceTransformers (RAG), and Coqui VITS (TTS). [web:20][web:159][web:50]

* Dual‑mode interaction:

  - Chat mode for open‑domain Q&A and explanations

 - Translation mode for speech‑to‑speech translation into a dynamically chosen target language

* Voice‑controlled mode & language switching:

 - “Translate into Spanish/French/German/…”

 - “Stop translation mode” / “Stop translation”

 - “What language are you translating into?”

* Stateful routing: explicit mode and current_target_lang maintained in the main loop, so every utterance is routed correctly between chat and translation.

* Robust microphone handling: queue‑flush and fixed ~6‑second recording windows to avoid the system re‑transcribing its own TTS output. [web:239]

* Clean separation of concerns across:

 - ASRPipeline

 - RAGPipeline

 - TranslatorPipeline

 - TTSPipeline

 - VoiceRAGTranslatorApp (orchestration + state machine)

### Setup & Run

* Prerequisites

- macOS with Python 3.10+

- Ollama installed and on PATH

- mistral model pulled in Ollama

1. Install and pull model
# Install Ollama (see official docs if not installed)
# Then pull Mistral locally
ollama pull mistral

2. Clone this repo

git clone https://github.com/<your-username>/local-offline-voice-translator-rag-assistant.git
cd local-offline-voice-translator-rag-assistant

3. Create and activate venu

python3 -m venv .venv
source .venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

5. start ollama server

ollama serve

6. Run the voice assistant

cd /Users/akilalourdes   # or this repo directory
source .venv/bin/activate
python main.py

[ASR] Loading Whisper model ...
[RAG] In-memory index ready
[TTS] TTS model loaded
[AUDIO] Microphone stream started
[APP] Voice chat + translator running. Press Ctrl+C to stop.


### Demo
Chat mode examples

Speak:

- “What is ASR?”

- “What is an LLM?”

- “How does this local voice assistant work?”

- “What can you do?”

The system will:

- Transcribe your question

- Retrieve a small RAG context describing Whisper / Ollama / Coqui

- Generate an answer via the local mistral model

- Speak the answer using Coqui TTS


Translation mode examples

1. Start in chat (default), then say:

-  “Translate into Spanish.”

-  Then: “Good morning, how are you?”

-  Then: “Thank you for your help.”

Example outputs:

-  “Buenos días, ¿cómo estás?”

-  “Gracias por tu ayuda.”


2. Switch to French:

-  “Translate into French.”

-  “I live in New York City.”

Example output:

- “J’habite à New York.”


3.Switch to German:

- “Translate into German.”

- “What is the climate now?”

Example output:

- “Welche Wetterlage herrscht derzeit?”


4. Stop translating:

- “Stop translation mode.” or “Stop translation.”

System returns to chat mode and answers normally.

###Metrics & Limitations

1. Latency

Measured on a Mac (Apple Silicon) for short (~3s) utterances:

- ASR + LLM + TTS end‑to‑end: ≈2–3 seconds from end of speech to audio playback, using faster-whisper “small” on CPU and mistral via Ollama. [web:245][web:227]

- This is acceptable for a local prototype without GPU; a production system could reduce latency with streaming ASR/TTS or smaller/quantized models.

2. Quality

i. ASR (Whisper small):

- Good for conversational English.

- Occasionally mishears unusual phrases or long, run‑on sentences. [web:159][web:306]

ii. Chat responses:

- RAG gives the model context about the pipeline components, so explanations of ASR/LLM/TTS/RAG are generally accurate.

- For open‑domain questions (weather, etc.), the model correctly notes when real‑time data is not available.

iii. Translation:

- English → Spanish/French/German: generally correct and natural for short, simple sentences (greetings, “I live in …”, “What is the climate now?”). [web:288][web:299]

- English → Indian languages (Tamil/Telugu/Kannada/Malayalam): experimental. A few demo phrases are hand‑checked; other translations may be inaccurate because the translator uses a general LLM as MT instead of a dedicated translation model. [web:20][web:327]

3. Current Limitations and Future Work

i. Utterance‑level, not fully streaming:

- The system records ~6 seconds, then processes; it does not yet perform true streaming ASR and TTS. [web:307]

ii. No diarization / speaker separation:

- Assumes a single active speaker at a time. [web:327]

iii. LLM‑based translation:

- For production‑grade quality, especially for Indic languages, this design would be extended to call dedicated MT models or services, and evaluated using BLEU/COMET and human review. [web:325]

iv. English‑only TTS:

- Current TTS voice is English (VCTK); a multilingual TTS model (e.g., XTTS) could better match non‑English outputs.

Despite these limitations, the project demonstrates a realistic end‑to‑end architecture for a local voice assistant and translator, including mic streaming, ASR, RAG, LLM orchestration, TTS, and stateful mode control.

