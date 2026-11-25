"""Command line interface: `python -m marcusrag index|ask|eval|info`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .citations import format_evidence
from .config import Settings, load_settings
from .pipeline import answer_question, build_index, index_status


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="YAML configuration file")
    common.add_argument("--data-dir", default=None, help="directory containing input documents")
    common.add_argument("--index-root", default=None, help="root directory for indexes")
    common.add_argument("--store", choices=["chroma", "memory"], default=None)
    common.add_argument("--embedder", choices=["hf", "hash"], default=None)
    common.add_argument("--embedding-model", default=None)
    common.add_argument("--profile", choices=["en", "zh"], default=None)
    common.add_argument("--chunk-size", type=int, default=None)
    common.add_argument("--chunk-overlap", type=int, default=None)
    common.add_argument("--top-k", type=int, default=None)
    common.add_argument("--fetch-k", type=int, default=None)
    common.add_argument("--search-type", choices=["hybrid", "mmr", "similarity"], default=None)
    common.add_argument("--rerank", choices=["none", "lexical", "cross-encoder"], default=None)
    common.add_argument("--rerank-model", default=None)
    common.add_argument(
        "--rerank-base-weight",
        type=float,
        default=None,
        help="lexical re-rank: share of the first-stage score kept (0..1)",
    )
    common.add_argument("--llm-backend", choices=["ollama", "fake"], default=None)
    common.add_argument("--llm-model", default=None)
    common.add_argument("--json", action="store_true", dest="as_json", help="machine readable output")
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="marcusrag",
        description="Evidence-based RAG over local documents (PDF / TXT / Markdown).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_cmd = sub.add_parser("index", parents=[common], help="build or update the vector index")
    index_cmd.add_argument("--rebuild", action="store_true", help="drop the existing index first")
    index_cmd.add_argument("--full", action="store_true", help="disable incremental skipping")
    index_cmd.add_argument("--dry-run", action="store_true", help="load and chunk only, write nothing")

    ask_cmd = sub.add_parser("ask", parents=[common], help="ask a question against the index")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--show-context", action="store_true", help="print the retrieved evidence")
    ask_cmd.add_argument("--color", action="store_true", help="highlight query terms in the evidence")

    eval_cmd = sub.add_parser("eval", parents=[common], help="run a JSONL evaluation set")
    eval_cmd.add_argument("--evalset", default="evals/qa.jsonl", help="JSONL evaluation cases")
    eval_cmd.add_argument("--k", type=int, default=None, help="cut-off for Recall@k / Hit@k / MRR")
    eval_cmd.add_argument(
        "--end-to-end",
        action="store_true",
        dest="end_to_end",
        help="also generate answers and measure citation precision / refusal accuracy",
    )
    eval_cmd.add_argument("--output", default=None, help="write the JSON report to this file")

    sub.add_parser("info", parents=[common], help="show configuration and index status")
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    overrides = {
        "data_dir": args.data_dir,
        "index_root": args.index_root,
        "store": args.store,
        "embedder": args.embedder,
        "embedding_model": args.embedding_model,
        "profile": args.profile,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
        "fetch_k": args.fetch_k,
        "search_type": args.search_type,
        "rerank": args.rerank,
        "rerank_model": args.rerank_model,
        "rerank_base_weight": args.rerank_base_weight,
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
    }
    return load_settings(args.config, **overrides)


def _print_warnings(warnings: Sequence[str]) -> None:
    for warning in warnings:
        print(f"[warn] {warning}")


def _cmd_index(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    _print_warnings(settings.warnings())

    if args.dry_run:
        from .chunking import split_documents
        from .documents import discover_files, scan_documents

        scanned = scan_documents(settings.data_dir, verbose=True)
        docs = scanned.docs
        chunks = split_documents(
            docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            profile=settings.profile,
        )
        payload = {
            "files": [p.name for p in discover_files(settings.data_dir)],
            "documents": len(docs),
            "chunks": len(chunks),
            "empty_sources": [{"file": name, "reason": reason} for name, reason in scanned.empty_sources],
            "dry_run": True,
        }
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"dry-run: {len(docs)} document segments -> {len(chunks)} chunks；未写入任何索引")
            for name, reason in scanned.empty_sources:
                print(f"  - {name}: {reason}")
        return 0

    report = build_index(
        settings,
        rebuild=args.rebuild,
        incremental=not args.full,
        verbose=not args.as_json,
    )
    payload = {
        "files": report.files,
        "documents": report.documents,
        "chunks": report.chunks,
        "written": report.written,
        "skipped_files": report.skipped_files,
        "index_dir": report.index_dir,
        "store": report.store,
        "per_source": report.per_source,
        "empty_sources": [{"file": name, "reason": reason} for name, reason in report.empty_sources],
        "warnings": report.warnings,
    }
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"files scanned     : {report.files} (skipped unchanged: {report.skipped_files})")
        print(f"document segments : {report.documents}")
        print(f"chunks produced   : {report.chunks}")
        print(f"vectors written   : {report.written}")
        print(f"index directory   : {report.index_dir} [{report.store}]")
        for name, count in sorted(report.per_source.items()):
            print(f"  - {name}: {count} chunk(s)")
        if report.written == 0:
            print("no new content; index is up to date")
        if report.empty_sources:
            print("")
            print(f"warning: {len(report.empty_sources)} file(s) contributed no text and were not indexed:")
            for name, reason in report.empty_sources:
                print(f"  - {name}: {reason}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    result = answer_question(settings, args.question)

    if args.as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        if args.show_context:
            print("--- retrieved evidence ---")
            print(format_evidence(result.evidence, color=args.color, query=args.question))
            print("--- answer ---")
        print(result.highlighted())
        if result.unverified_citations:
            print("\n[warn] 以下引用未出现在本次召回证据中，可能是模型编造：")
            for citation in result.unverified_citations:
                print(f"  - [{citation.file}:{citation.page}]")

    if result.refused:
        if not args.as_json:
            if result.evidence:
                print(
                    "\n[hint] 已召回证据，但回答没有引用其中任何一条。"
                    "换更强的生成模型，或确认 --llm-backend 不是 fake（fake 只产出固定桩文本）。"
                )
            else:
                print("\n[hint] 未找到证据。请先运行 `python -m marcusrag index`，或换个问法。")
        return 2
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate

    settings = _settings_from_args(args)
    _print_warnings(settings.warnings())

    llm = None
    if args.end_to_end:
        from .llm import build_llm

        llm = build_llm(settings.llm_backend, settings.llm_model)

    report = evaluate(settings, args.evalset, llm=llm, k=args.k)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.format())
        if args.output:
            print(f"report written   : {args.output}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    status = index_status(settings)
    if args.as_json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    print("configuration")
    print(f"  data_dir        : {status['data_dir']}")
    print(f"  index_dir       : {status['index_dir']}")
    print(f"  store           : {status['store']}")
    print(f"  embedding_model : {status['embedding_model']}")
    print(f"  chunk           : {settings.chunk_size}/{settings.chunk_overlap} (profile={settings.profile})")
    print(f"  llm             : {settings.llm_backend}:{settings.llm_model}")
    print("index")
    print(f"  indexed files   : {status['manifest_files']}")
    print(f"  indexed chunks  : {status['manifest_chunks']}")
    print(f"  input documents : {', '.join(status['data_files']) or '(none)'}")
    _print_warnings(status["warnings"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"index": _cmd_index, "ask": _cmd_ask, "eval": _cmd_eval, "info": _cmd_info}
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ValueError, ModuleNotFoundError, ConnectionError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
