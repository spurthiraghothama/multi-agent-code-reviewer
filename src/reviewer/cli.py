"""Command-line entry points, handy for local development.

    python -m reviewer.cli review owner/repo 42 [--dry-run]   # review a PR without a webhook
    python -m reviewer.cli index /path/to/repo owner/repo     # (re)build the vector index
    python -m reviewer.cli graph /path/to/repo module.func    # inspect callers from the call graph
    python -m reviewer.cli models                             # list Gemini models your key can use
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from reviewer.config import get_settings


def main() -> None:
    p = argparse.ArgumentParser(prog="reviewer")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review")
    r.add_argument("repo")
    r.add_argument("pr", type=int)
    r.add_argument("--dry-run", action="store_true")
    i = sub.add_parser("index")
    i.add_argument("path", type=Path)
    i.add_argument("repo")
    g = sub.add_parser("graph")
    g.add_argument("path", type=Path)
    g.add_argument("function")
    sub.add_parser("models")
    args = p.parse_args()

    settings = get_settings()
    if args.cmd == "review":
        from reviewer.main import run_review
        if args.dry_run:
            settings.dry_run = True
        asyncio.run(run_review(args.repo, args.pr))
    elif args.cmd == "index":
        from reviewer.indexing.parser import parse_repo
        from reviewer.indexing.store import CodeIndex
        print(CodeIndex(settings.chroma_path, args.repo).sync(parse_repo(args.path)))
    elif args.cmd == "graph":
        from reviewer.indexing.call_graph import CallGraph
        from reviewer.indexing.parser import parse_repo
        cg = CallGraph(parse_repo(args.path))
        for e in cg.callers(args.function):
            c = cg.chunks[e.caller]
            print(f"{e.kind:5}  {e.caller}  ({c.file}:{c.start_line})")
    elif args.cmd == "models":
        from google import genai

        from reviewer.agents.llm import GeminiLLM
        GeminiLLM(settings)  # validates that a key is configured, with a helpful message if not
        client = genai.Client(api_key=settings.gemini_api_key or None)
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print(m.name.removeprefix("models/"))


if __name__ == "__main__":
    main()
