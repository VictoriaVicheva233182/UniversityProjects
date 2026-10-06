"""The assistant pipeline: guard input, retrieve, look up account, generate, guard output."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from assurance_lab.config import AppConfig
from assurance_lab.guardrails.input_classifier import InputClassifier
from assurance_lab.guardrails.output_filter import OutputFilter
from assurance_lab.guardrails.sanitizer import sanitize_untrusted
from assurance_lab.llm.base import LLMClient
from assurance_lab.rag.customers import AccountLookupTool, CustomerStore
from assurance_lab.rag.documents import Chunk
from assurance_lab.rag.prompts import SAFE_REFUSAL, build_messages
from assurance_lab.rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievedSource:
    id: str
    title: str
    access: str
    trust: str
    score: float
    action: str = "used"  # used | sanitized | dropped_by_classifier


@dataclass
class AssistantResponse:
    answer: str
    blocked: bool = False
    block_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    sources: list[RetrievedSource] = field(default_factory=list)
    customer_ids_in_context: list[str] = field(default_factory=list)
    input_score: float | None = None
    latency_ms: float = 0.0
    context: str = ""  # everything the model saw besides the system prompt

    def to_dict(self, include_context: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_context:
            data.pop("context")
        return data


class AssistantPipeline:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        retriever: Retriever,
        customers: CustomerStore,
        input_classifier: InputClassifier | None = None,
        output_filter: OutputFilter | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.retriever = retriever
        self.customers = customers
        self.lookup_tool = AccountLookupTool(customers, config.security.authorize_customer_lookup)
        self.input_classifier = input_classifier
        self.output_filter = output_filter

    # -- helpers -------------------------------------------------------------
    def _allow(self, chunk: Chunk) -> bool:
        return not (self.config.retrieval.enforce_access_control and chunk.access != "public")

    def _prepare_chunks(
        self, question: str, resp: AssistantResponse
    ) -> tuple[list, dict[str, str]]:
        sec = self.config.security
        hits = self.retriever.search(question, self.config.retrieval.top_k, allow=self._allow)
        kept, texts = [], {}
        for hit in hits:
            source = RetrievedSource(
                hit.chunk.id,
                hit.chunk.title,
                hit.chunk.access,
                hit.chunk.trust,
                round(hit.score, 4),
            )
            text = hit.chunk.text
            if hit.chunk.trust == "untrusted":
                if (
                    self.input_classifier is not None
                    and sec.input_classifier.screen_retrieved_chunks
                    and self.input_classifier.check(text).flagged
                ):
                    source.action = "dropped_by_classifier"
                    resp.sources.append(source)
                    resp.flags.append("untrusted_chunk_dropped")
                    continue
                if sec.sanitize_untrusted_context:
                    cleaned = sanitize_untrusted(text)
                    if cleaned.removed:
                        text = cleaned.text
                        source.action = "sanitized"
                        resp.flags.append("untrusted_chunk_sanitized")
            resp.sources.append(source)
            kept.append(hit)
            texts[hit.chunk.id] = text
        return kept, texts

    # -- main entry ----------------------------------------------------------
    def answer(self, question: str, session_customer_id: str) -> AssistantResponse:
        start = time.perf_counter()
        resp = AssistantResponse(answer="")

        if self.input_classifier is not None and self.config.security.input_classifier.enabled:
            decision = self.input_classifier.check(question)
            resp.input_score = round(decision.score, 4)
            if decision.flagged:
                resp.answer, resp.blocked, resp.block_reason = (
                    SAFE_REFUSAL,
                    True,
                    "input_classifier",
                )
                resp.flags.append("input_blocked")
                resp.latency_ms = round((time.perf_counter() - start) * 1000, 1)
                return resp

        chunks, chunk_texts = self._prepare_chunks(question, resp)
        lookup = self.lookup_tool.lookup(session_customer_id, question)
        if lookup.denied:
            resp.flags.append("account_lookup_denied")
        resp.customer_ids_in_context = [c.customer_id for c in lookup.customers]

        messages = build_messages(
            question,
            chunks,
            lookup.customers,
            hardened=self.config.security.hardened_prompt,
            chunk_texts=chunk_texts,
        )
        resp.context = messages[-1].content
        answer = self.llm.chat(messages).strip()

        if self.output_filter is not None:
            filtered = self.output_filter.apply(answer, session_customer_id, resp.context)
            answer = filtered.text
            resp.flags.extend(filtered.actions)
            if filtered.blocked:
                resp.blocked, resp.block_reason = True, filtered.actions[-1]

        resp.answer = answer
        resp.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return resp
