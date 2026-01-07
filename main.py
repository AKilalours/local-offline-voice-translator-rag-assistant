import os
import queue
import sys
import signal
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from TTS.api import TTS


# ----------------------------
# Config
# ----------------------------

@dataclass
class AppConfig:
    sample_rate: int = 16000
    block_size: int = 1024
    device: Optional[int] = None

    whisper_model_size: str = "small"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "mistral"  # must exist in `ollama list`[web:227]

    target_lang_code: str = "en"   # TTS model language (VCTK is English)


# ----------------------------
# Microphone streaming
# ----------------------------

class MicrophoneStream:
    def __init__(self, sample_rate: int, block_size: int, device=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self._queue = queue.Queue()
        self._stream = None

    def flush(self):
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"[AUDIO] Input status: {status}", file=sys.stderr)
        self._queue.put(indata.copy())

    def start(self):
        print("[AUDIO] Microphone stream started")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=self.device,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[AUDIO] Microphone stream stopped")

    def read_all(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        try:
            audio_chunks = []
            while True:
                chunk = self._queue.get(timeout=timeout)
                audio_chunks.append(chunk)
                # ~6 seconds per utterance
                if len(audio_chunks) * self.block_size >= self.sample_rate * 6:
                    break
            audio = np.concatenate(audio_chunks, axis=0)
            return audio.flatten()
        except queue.Empty:
            return None


# ----------------------------
# ASR with Faster Whisper
# ----------------------------

class ASRPipeline:
    def __init__(self, model_size: str, sample_rate: int):
        print("[ASR] Loading Whisper model (first time may take a while)...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")  # [web:159]
        self.sample_rate = sample_rate
        print("[ASR] Whisper model loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        segments, _ = self.model.transcribe(audio, language="en")
        text = "".join([seg.text for seg in segments]).strip()
        print(f"[ASR] Detected text: {text}")
        return text


# ----------------------------
# Simple RAG over small corpus
# ----------------------------

class RAGPipeline:
    def __init__(self, embed_model_name: str, ollama_url: str, ollama_model: str):
        print("[RAG] Loading SentenceTransformer embeddings...")
        self.embed_model = SentenceTransformer(embed_model_name)
        self.corpus = [
            "This is a local AI voice assistant with speech-to-text, RAG, and TTS.",
            "It can answer questions, translate text, and have conversations offline.",
            "The system uses Whisper for ASR, a local LLM via Ollama, and Coqui TTS.",
        ]
        self.corpus_embeddings = self.embed_model.encode(self.corpus, convert_to_numpy=True)
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        print("[RAG] In-memory index ready")

    def _retrieve(self, query: str, top_k: int = 2):
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.corpus_embeddings)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [self.corpus[i] for i in top_idx]

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(self.ollama_url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()  # expects "response"[web:227]
        return data.get("response", "").strip()

    def query(self, user_text: str) -> str:
        retrieved = self._retrieve(user_text, top_k=2)
        context = "\n".join(retrieved)
        prompt = (
            "You are a helpful AI voice assistant. Use the context to answer the user.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER: {user_text}\n\n"
            "ASSISTANT:"
        )
        print("[RAG] Querying local LLM via Ollama...")
        answer = self._call_ollama(prompt)
        print(f"[RAG] LLM answer: {answer}")
        return answer


# ----------------------------
# Translator using Ollama
# ----------------------------

class TranslatorPipeline:
    def __init__(self, ollama_url: str, ollama_model: str):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def translate(self, text: str, target_lang: str) -> str:
        prompt = (
            f"Translate the following text into {target_lang}.\n\n"
            f"Text: {text}\n\n"
            "Only output the translated text, nothing else."
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(self.ollama_url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()  # "response" holds translation[web:227]
        return data.get("response", "").strip()


# ----------------------------
# TTS with Coqui VITS (multi-speaker)
# ----------------------------

class TTSPipeline:
    def __init__(self, target_lang_code: str):
        print("[TTS] Loading TTS model...")
        self.tts = TTS(model_name="tts_models/en/vctk/vits")  # English VITS[web:131][web:50]
        self.target_lang_code = target_lang_code
        self.speaker_id = "p225"
        print("[TTS] TTS model loaded")

    def synthesize(self, text: str, out_path: str = "output_tts.wav"):
        print("[TTS] Synthesizing speech...")
        self.tts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker=self.speaker_id,
        )
        print(f"[TTS] Saved to {out_path}")
        data, sr = sf.read(out_path)
        sd.play(data, sr)
        sd.wait()


# ----------------------------
# Main app: chat + dynamic translation mode
# ----------------------------

class VoiceRAGTranslatorApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        self.mic = MicrophoneStream(
            sample_rate=cfg.sample_rate,
            block_size=cfg.block_size,
            device=cfg.device,
        )
        self.asr = ASRPipeline(cfg.whisper_model_size, cfg.sample_rate)
        self.rag = RAGPipeline(cfg.embed_model_name, cfg.ollama_url, cfg.ollama_model)
        self.translator = TranslatorPipeline(cfg.ollama_url, cfg.ollama_model)
        self.tts = TTSPipeline(cfg.target_lang_code)

        # Conversation state
        self.mode: str = "chat"          # "chat" or "translate"
        self.current_target_lang: str = "English"  # default target when translating

        self._running = True

    def stop(self):
        self._running = False
        self.mic.stop()

    # -------- Command helpers --------

    def _handle_mode_commands(self, text: str) -> bool:
        """
        Returns True if a mode/target command was handled (and we should return),
        False if no command matched.
        """
        t = text.strip().lower()

        # Turn translation mode on
        if "start translation mode" in t:
            self.mode = "translate"
            self.tts.synthesize(f"Translation mode on. I will translate into {self.current_target_lang}.")
            return True

        # Turn translation mode off: catch multiple phrasings
        if ("stop translation mode" in t
                or "exit translation mode" in t
                or "stop translation" in t):
            self.mode = "chat"
            self.tts.synthesize("Translation mode off. I will answer normally.")
            return True

        # Change target language: "translate into spanish", "translate to tamil", etc.
        if t.startswith("translate into ") or t.startswith("translate to "):
            parts = t.split()
            if len(parts) >= 3:
                # naive: last word is language
                target_lang_raw = parts[-1].rstrip(".,!?")
                self.current_target_lang = target_lang_raw.capitalize()
                self.mode = "translate"
                self.tts.synthesize(f"Okay, I will translate into {self.current_target_lang}.")
                return True

        # "What language are you translating into?"
        if "what language are you translating" in t:
            msg = f"I am translating into {self.current_target_lang}."
            self.tts.synthesize(msg)
            return True

        return False

    # -------- Main loop step --------

    def run_once(self):
        print("\n[APP] Speak now (Ctrl+C to exit). Listening...")

        self.mic.flush()
        audio = self.mic.read_all(timeout=2.0)
        if audio is None:
            print("[APP] No speech detected, try again.")
            return

        asr_text = self.asr.transcribe(audio, self.cfg.sample_rate)
        if not asr_text:
            print("[APP] Empty transcription.")
            return

        text = asr_text.strip()

        # 1) Try to handle mode / target-language control commands first
        if self._handle_mode_commands(text):
            return

        # 2) Route by mode
        if self.mode == "translate":
            print(f"[APP] Source text: {text}")
            print(f"[APP] Translating to: {self.current_target_lang}")
            translated = self.translator.translate(text, self.current_target_lang)
            print(f"[APP] Translated text: {translated}")
            self.tts.synthesize(translated)
        else:
            # chat mode
            answer = self.rag.query(text)
            self.tts.synthesize(answer)

    def run_loop(self):
        self.mic.start()
        print("[APP] Voice chat + translator running. Press Ctrl+C to stop.")
        try:
            while self._running:
                self.run_once()
        except KeyboardInterrupt:
            print("\n[APP] Keyboard interrupt, stopping...")
        finally:
            self.stop()


# ----------------------------
# Entry point
# ----------------------------

def main():
    cfg = AppConfig()
    app = VoiceRAGTranslatorApp(cfg)

    def handle_sigint(sig, frame):
        print("\n[APP] Caught SIGINT, shutting down...")
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    app.run_loop()


if __name__ == "__main__":
    main()
