"""Wire configuration into a ready-to-use assistant pipeline."""

from __future__ import annotations

import logging

from assurance_lab.config import AppConfig, resolve_path
from assurance_lab.guardrails.input_classifier import InputClassifier
from assurance_lab.guardrails.output_filter import OutputFilter
from assurance_lab.llm.base import LLMClient
from assurance_lab.llm.factory import create_llm
from assurance_lab.rag.customers import CustomerStore
from assurance_lab.rag.documents import load_corpus
from assurance_lab.rag.pipeline import AssistantPipeline
from assurance_lab.rag.prompts import SYSTEM_CANARY
from assurance_lab.rag.retriever import TfidfRetriever

logger = logging.getLogger(__name__)

# Canary planted in the confidential playbook (data/knowledge_base/internal_fraud_playbook.md).
DOCUMENT_CANARY = "HKB-CONF-4471"


def build_pipeline(config: AppConfig, llm: LLMClient | None = None) -> AssistantPipeline:
    dirs = [resolve_path(config.data.knowledge_base_dir)]
    if config.data.untrusted_sources_dir is not None:
        dirs.append(resolve_path(config.data.untrusted_sources_dir))
    chunks = load_corpus(dirs, config.retrieval.chunk_max_chars)
    customers = CustomerStore.from_json(resolve_path(config.data.customers_file))

    sec = config.security
    classifier = None
    if sec.input_classifier.enabled or sec.input_classifier.screen_retrieved_chunks:
        classifier = InputClassifier.load(
            resolve_path(sec.input_classifier.model_path), sec.input_classifier.threshold
        )

    output_filter = None
    if sec.output_filter.enabled:
        output_filter = OutputFilter(
            canaries=[SYSTEM_CANARY, DOCUMENT_CANARY],
            customers=customers,
            allowed_domains=sec.output_filter.allowed_domains,
            grounding_check=sec.output_filter.grounding_check,
        )

    pipeline = AssistantPipeline(
        config=config,
        llm=llm or create_llm(config.llm),
        retriever=TfidfRetriever(chunks),
        customers=customers,
        input_classifier=classifier,
        output_filter=output_filter,
    )
    logger.info(
        "Pipeline ready: profile=%s llm=%s chunks=%d",
        config.profile,
        pipeline.llm.describe(),
        len(chunks),
    )
    return pipeline
