# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel, Field

from main import (
    AppConfig,
    RAGPipeline,
    TranslatorPipeline,
    GeneralLLMPipeline,
    REFUSAL,
    _normalize_role,
    _is_live_info_query,
    _mentions_project_terms,
)

app = FastAPI(title="Offline Voice Translator + RAG API")

# Initialize pipelines once (server startup)
cfg = AppConfig(
    retrieval_backend="dense_rerank",
    log_level="demo",
    show_chunk_text=False,
    max_chunks_to_print=4,
    enable_general_fallback=True,
)
rag = RAGPipeline(cfg)
translator = TranslatorPipeline(cfg)
general = GeneralLLMPipeline(cfg)


class AskRequest(BaseModel):
    text: str = Field(..., min_length=1)
    role: str = Field(default="public", description="public|internal|confidential|admin")
    strict: bool = Field(default=False, description="If true, do not fall back to general LLM.")


class AskResponse(BaseModel):
    answer: str
    refused: bool
    role: str


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_lang: str = Field(..., min_length=2)


class TranslateResponse(BaseModel):
    translated: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    role = _normalize_role(req.role)
    text = req.text.strip()

    # Cleaner UX for live queries
    if _is_live_info_query(text):
        msg = (
            "I run offline, so I can’t fetch live news, weather, or market prices. "
            "If you want real-time results, you’d need to connect an online API."
        )
        return AskResponse(answer=msg, refused=True, role=role)

    ans = rag.query(text, user_role=role)
    refused = (ans.strip() == REFUSAL)

    # Optional fallback
    if refused and (not req.strict) and cfg.enable_general_fallback and (not _mentions_project_terms(text)):
        ans = general.query(text)
        refused = (ans.strip() == REFUSAL)

    return AskResponse(answer=ans, refused=refused, role=role)


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    translated = translator.translate(req.text.strip(), req.target_lang.strip())
    return TranslateResponse(translated=translated)
