"""ChromaDB vector store with incremental (hash-based) re-indexing.

Embedding is the expensive part of indexing, so we keep a manifest {file: sha256(content)} per
repo and only re-embed files whose content changed since the last sync. On a typical PR that's
just the handful of files the PR touches; the first PR on a repo pays for a full index.

Embeddings: Chroma's default embedding function (all-MiniLM-L6-v2 via ONNX, runs locally on CPU).
Trade-off: a code-specific embedding model would retrieve better, but this one is free, needs no
API key, and is good enough for "find code that looks like this" - the call graph does the
precise structural work, embeddings only supply the fuzzy "related code" signal.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import defaultdict
from pathlib import Path

import chromadb

from reviewer.indexing.parser import CodeChunk, ModuleInfo

log = logging.getLogger(__name__)

_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
MAX_EMBED_CHARS = 4000


def _collection_name(repo_full_name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_full_name)[:60]
    return name if len(name) >= 3 else f"{name}___"


def chunk_document(c: CodeChunk) -> str:
    # Include path + name: a lot of semantic signal in code lives in identifiers.
    return f"# {c.file} :: {c.qualname}\n{c.source[:MAX_EMBED_CHARS]}"


class CodeIndex:
    def __init__(self, chroma_path: Path, repo_full_name: str, client: chromadb.ClientAPI | None = None):
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = client or chromadb.PersistentClient(path=str(chroma_path))
        self.repo = repo_full_name
        self.collection = self.client.get_or_create_collection(_collection_name(repo_full_name),
                                                               metadata={"hnsw:space": "cosine"})
        self.manifest_path = chroma_path / f"{_collection_name(repo_full_name)}.manifest.json"

    def _load_manifest(self) -> dict[str, str]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def sync(self, modules: dict[str, ModuleInfo]) -> dict[str, int]:
        """Bring the index in line with `modules`. Returns counts for logging/metrics."""
        # A per-repo lock: two PRs on the same repo arriving together must not interleave
        # delete/add on the same collection.
        with _locks[self.repo]:
            manifest = self._load_manifest()
            changed = [f for f, m in modules.items() if manifest.get(f) != m.file_hash]
            removed = [f for f in manifest if f not in modules]

            for f in changed + removed:
                self.collection.delete(where={"file": f})

            chunks = [c for f in changed for c in modules[f].chunks]
            for i in range(0, len(chunks), 256):  # batch to keep memory bounded
                batch = chunks[i:i + 256]
                self.collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    documents=[chunk_document(c) for c in batch],
                    metadatas=[{"file": c.file, "qualname": c.qualname, "kind": c.kind,
                                "start_line": c.start_line, "end_line": c.end_line} for c in batch],
                )

            for f in removed:
                manifest.pop(f, None)
            for f in changed:
                manifest[f] = modules[f].file_hash
            self.manifest_path.write_text(json.dumps(manifest))
        stats = {"files_reindexed": len(changed), "files_removed": len(removed), "chunks_embedded": len(chunks)}
        log.info("Index sync for %s: %s", self.repo, stats)
        return stats

    def similar(self, chunk: CodeChunk, k: int, exclude_ids: set[str]) -> list[dict]:
        """Top-k semantically similar chunks, excluding the chunk itself and anything already in context."""
        if self.collection.count() == 0:
            return []
        res = self.collection.query(query_texts=[chunk_document(chunk)],
                                    n_results=min(k + len(exclude_ids) + 1, self.collection.count()))
        out = []
        for cid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
            if cid == chunk.chunk_id or cid in exclude_ids:
                continue
            out.append({"id": cid, "distance": dist, **meta})
            if len(out) >= k:
                break
        return out
