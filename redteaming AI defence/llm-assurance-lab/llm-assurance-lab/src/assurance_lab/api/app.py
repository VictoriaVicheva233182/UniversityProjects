"""HTTP API for the assistant.

Run with ``assurance-lab serve`` and open http://127.0.0.1:8000 for the test bench.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from assurance_lab import __version__
from assurance_lab.config import load_config, resolve_path
from assurance_lab.factory import build_pipeline
from assurance_lab.guardrails.input_classifier import GuardrailUnavailable
from assurance_lab.llm.base import LLMError
from assurance_lab.rag.customers import CustomerStore
from assurance_lab.rag.pipeline import AssistantPipeline

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
Profile = Literal["baseline", "hardened"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    profile: Profile = "baseline"
    customer_id: str = Field(default="C1001", pattern=r"^C\d{4}$")


class ChatResponse(BaseModel):
    answer: str
    blocked: bool
    block_reason: str | None
    flags: list[str]
    sources: list[dict[str, Any]]
    customer_ids_in_context: list[str]
    input_score: float | None
    latency_ms: float
    profile: Profile


class PipelineRegistry:
    """Builds each profile once, lazily, and reuses it across requests."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self._pipelines: dict[str, AssistantPipeline] = {}
        self._lock = threading.Lock()

    def get(self, profile: str) -> AssistantPipeline:
        with self._lock:
            if profile not in self._pipelines:
                cfg = load_config(self.config_dir / f"{profile}.yaml")
                self._pipelines[profile] = build_pipeline(cfg)
            return self._pipelines[profile]


def create_app(config_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Havenkade assistant, LLM Assurance Lab", version=__version__)
    registry = PipelineRegistry(config_dir or resolve_path("configs"))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/customers")
    def customers() -> list[dict[str, str]]:
        cfg = load_config(registry.config_dir / "baseline.yaml")
        store = CustomerStore.from_json(resolve_path(cfg.data.customers_file))
        return [{"customer_id": c.customer_id, "name": c.name} for c in store]

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        try:
            pipeline = registry.get(req.profile)
            resp = pipeline.answer(req.message, req.customer_id)
        except GuardrailUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        data = resp.to_dict()
        return ChatResponse(profile=req.profile, **data)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
