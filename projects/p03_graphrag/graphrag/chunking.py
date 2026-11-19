"""Sentence-aligned, token-budgeted document chunking."""
from __future__ import annotations

from aikit import count_tokens, truncate_to_tokens

from .models import Chunk, Document
from .text import split_sentences


class SentenceChunker:
    """Pack whole sentences into chunks of at most ``max_tokens`` tokens.

    ``overlap_sentences`` trailing sentences of each chunk are repeated at the
    start of the next one so facts spanning a boundary keep their context.
    A single sentence longer than the budget is truncated rather than dropped.
    """

    def __init__(self, max_tokens: int = 160, overlap_sentences: int = 0):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        fresh = 0  # sentences in `current` not carried over as overlap

        def flush() -> None:
            text = " ".join(current)
            chunks.append(
                Chunk(
                    id=f"{doc.id}#{len(chunks)}",
                    doc_id=doc.id,
                    index=len(chunks),
                    text=text,
                    n_tokens=count_tokens(text),
                )
            )

        for sent in sentences:
            cost = count_tokens(sent)
            if cost > self.max_tokens:
                sent = truncate_to_tokens(sent, self.max_tokens)
                cost = count_tokens(sent)
            if current and current_tokens + cost + 1 > self.max_tokens and fresh > 0:
                flush()
                carry = current[-self.overlap_sentences :] if self.overlap_sentences else []
                current = list(carry)
                current_tokens = count_tokens(" ".join(current)) if current else 0
                fresh = 0
                # Drop overlap that no longer leaves room for the new sentence.
                while current and current_tokens + cost + 1 > self.max_tokens:
                    current.pop(0)
                    current_tokens = count_tokens(" ".join(current)) if current else 0
            current.append(sent)
            current_tokens = count_tokens(" ".join(current))
            fresh += 1
        if current and fresh > 0:
            flush()
        return chunks

    def chunk_all(self, docs: list[Document]) -> list[Chunk]:
        return [c for d in docs for c in self.chunk(d)]
