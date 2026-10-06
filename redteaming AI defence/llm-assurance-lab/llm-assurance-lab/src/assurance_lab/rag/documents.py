"""Load markdown documents with YAML front matter and split them into chunks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

logger = logging.getLogger(__name__)

Access = Literal["public", "internal"]
Trust = Literal["trusted", "untrusted"]


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    heading: str
    text: str
    access: Access
    trust: Trust
    source: str

    @property
    def search_text(self) -> str:
        return f"{self.title}. {self.heading}. {self.text}"


def _parse_front_matter(raw: str, path: Path) -> tuple[dict[str, Any], str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.DOTALL)
    if not m:
        raise ValueError(f"{path} has no YAML front matter")
    meta = yaml.safe_load(m.group(1)) or {}
    for key in ("id", "title", "access", "trust"):
        if key not in meta:
            raise ValueError(f"{path} front matter is missing '{key}'")
    if meta["access"] not in ("public", "internal"):
        raise ValueError(f"{path}: access must be public or internal")
    if meta["trust"] not in ("trusted", "untrusted"):
        raise ValueError(f"{path}: trust must be trusted or untrusted")
    return meta, m.group(2)


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current.strip())
            current = ""
        current += sentence + " "
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_document(path: Path, max_chars: int = 700) -> list[Chunk]:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"), path)
    sections = re.split(r"^##\s+", body, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        heading, _, text = section.partition("\n")
        text = " ".join(text.split())
        if not text:
            continue
        for piece in _split_long(text, max_chars):
            chunks.append(
                Chunk(
                    id=f"{meta['id']}#{len(chunks) + 1}",
                    doc_id=str(meta["id"]),
                    title=str(meta["title"]),
                    heading=heading.strip(),
                    text=piece,
                    access=cast(Access, meta["access"]),
                    trust=cast(Trust, meta["trust"]),
                    source=str(meta.get("source", path.name)),
                )
            )
    return chunks


def load_corpus(dirs: list[Path], max_chars: int = 700) -> list[Chunk]:
    chunks: list[Chunk] = []
    for directory in dirs:
        if not directory.is_dir():
            logger.warning("Document directory not found, skipping: %s", directory)
            continue
        for path in sorted(directory.glob("*.md")):
            chunks.extend(chunk_document(path, max_chars))
    if not chunks:
        raise ValueError(f"No documents found in {', '.join(map(str, dirs))}")
    logger.debug("Loaded %d chunks", len(chunks))
    return chunks
