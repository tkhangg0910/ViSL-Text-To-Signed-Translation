#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visl_cli.py — Command-line interface for ViSL (Vietnamese Sign Language) Translator

Usage examples:
  # Basic (single sentence)
  python visl_cli.py "Hôm nay trời nắng đẹp."

  # Specify dialect and output path
  python visl_cli.py "Bạn tên gì?" --dialect "🇸 South" --output result.mp4

  # Ensemble mode
  python visl_cli.py "Tôi muốn học ngôn ngữ ký hiệu." --ensemble

  # Batch mode (one sentence per line from a text file)
  python visl_cli.py --batch sentences.txt --output-dir ./outputs/

  # Interactive (REPL) mode
  python visl_cli.py --interactive
"""

import os
import sys
import json
import argparse
import threading
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="visl_cli",
        description="ViSL — Vietnamese Sign Language Translator (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "sentence",
        nargs="?",
        help="Vietnamese sentence to translate (positional argument).",
    )
    input_group.add_argument(
        "--batch", "-b",
        metavar="FILE",
        help="Path to a text file with one sentence per line.",
    )
    input_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive REPL session.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        default=None,
        help="Output video path (.mp4). Defaults to ./outputs/<slug>.mp4",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="./outputs",
        help="Directory for output videos in batch/interactive mode (default: ./outputs).",
    )

    # ── Dialect ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dialect", "-d",
        choices=["north", "central", "south"],
        default="south",
        help="Regional dialect: north / central / south (default: south).",
    )

    # ── Model / index config ──────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ensemble",
        action="store_true",
        help="Run base + large models in parallel and merge results.",
    )
    mode_group.add_argument(
        "--advanced",
        action="store_true",
        help="Use fully custom paths (requires --embedding-model, --index, --meta).",
    )

    parser.add_argument(
        "--model-type",
        default="large",
        choices=["base", "large"],
        help="Embedding model variant in single mode (default: large).",
    )
    parser.add_argument(
        "--model-type-2",
        default=None,
        help="Second model type for ensemble (default: the other of base/large).",
    )
    parser.add_argument("--poses-path",      default="./poses/")
    parser.add_argument("--embedding-model", default=None, metavar="PATH",
                        help="HuggingFace model path/id (advanced mode).")
    parser.add_argument("--index",           default=None, metavar="PATH",
                        help="FAISS index path (advanced mode).")
    parser.add_argument("--meta",            default=None, metavar="PATH",
                        help="Metadata JSON path (advanced mode).")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates retrieved per token (default: 5).",
    )

    # ── Verbosity ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print intermediate pipeline outputs (gloss, tokens, retrievals).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print pipeline result as JSON to stdout (useful for scripting).",
    )

    return parser.parse_args()


# ── Dialect key mapping ───────────────────────────────────────────────────────

DIALECT_KEYS = {
    "north":   "🇳 North",
    "central": "🇨 Central",
    "south":   "🇸 South",
}

DIALECT_DB_CODES = {
    "north":   "AB",
    "central": "AT",
    "south":   "AN",
}


# ── Build dialect config (mirrors app.py logic) ───────────────────────────────

def build_dialect_config(args):
    dialect_key = DIALECT_KEYS[args.dialect]

    if args.advanced:
        if not all([args.embedding_model, args.index, args.meta]):
            sys.exit(
                "❌  --advanced mode requires --embedding-model, --index, and --meta."
            )
        embedding_model_path   = args.embedding_model
        embedding_model_path_2 = None
        dialect_config = {
            dialect_key: {
                "faiss_index":   args.index,
                "metadata":      args.meta,
                "faiss_index_2": None,
                "metadata_2":    None,
            }
        }

    elif args.ensemble:
        mt1 = args.model_type
        mt2 = args.model_type_2 or ("base" if mt1 == "large" else "large")
        embedding_model_path   = f"tkhangg0910/viconbert-{mt1}"
        embedding_model_path_2 = f"tkhangg0910/viconbert-{mt2}"
        db_code = DIALECT_DB_CODES[args.dialect]
        dialect_config = {
            dialect_key: {
                "faiss_index":   f"./pose_databases/index_mean_{db_code}_{mt1}.faiss",
                "metadata":      f"./pose_databases/metadata_mean_{db_code}_{mt1}.json",
                "faiss_index_2": f"./pose_databases/index_mean_{db_code}_{mt2}.faiss",
                "metadata_2":    f"./pose_databases/metadata_mean_{db_code}_{mt2}.json",
            }
        }
        print(f"[Ensemble] Model-1: {embedding_model_path}")
        print(f"[Ensemble] Model-2: {embedding_model_path_2}")

    else:  # basic
        mt = args.model_type
        embedding_model_path   = f"tkhangg0910/viconbert-{mt}"
        embedding_model_path_2 = None
        db_code = DIALECT_DB_CODES[args.dialect]
        dialect_config = {
            dialect_key: {
                "faiss_index":   f"./pose_databases/index_mean_{db_code}_{mt}.faiss",
                "metadata":      f"./pose_databases/metadata_mean_{db_code}_{mt}.json",
                "faiss_index_2": None,
                "metadata_2":    None,
            }
        }

    return dialect_key, dialect_config, embedding_model_path, embedding_model_path_2


# ── Load models ───────────────────────────────────────────────────────────────

def load_models(embedding_model_path, embedding_model_path_2, ensemble):
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CACHE_DIR = "./models_cache"

    def _load(path, label):
        print(f"⏳ Loading {label}: {path} ...")
        model = AutoModel.from_pretrained(
            path, trust_remote_code=True, ignore_mismatched_sizes=True,
            cache_dir=CACHE_DIR,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, cache_dir=CACHE_DIR)
        print(f"✅ {label} loaded on {device}")
        return model, tokenizer

    if ensemble:
        m1, t1, m2, t2 = None, None, None, None

        def _l1():
            nonlocal m1, t1
            m1, t1 = _load(embedding_model_path, "Model-1")

        def _l2():
            nonlocal m2, t2
            m2, t2 = _load(embedding_model_path_2, "Model-2")

        th1 = threading.Thread(target=_l1, daemon=True)
        th2 = threading.Thread(target=_l2, daemon=True)
        th1.start(); th2.start()
        th1.join();  th2.join()
        return m1, t1, m2, t2
    else:
        m, t = _load(embedding_model_path, "Embedding model")
        return m, t, None, None


# ── Build pipeline ────────────────────────────────────────────────────────────

def build_pipeline(args, dialect_key, dialect_config,
                   emb_model, emb_tokenizer, emb_model_2, emb_tokenizer_2):
    from visl_pipeline import ViSLPipeline

    cfg = dialect_config[dialect_key]
    print(f"⏳ Initializing pipeline for dialect: {dialect_key} ...")
    pipeline = ViSLPipeline(
        poses_path=args.poses_path,
        embedding_model=emb_model,
        embedding_tokenizer=emb_tokenizer,
        faiss_index_path=cfg["faiss_index"],
        metadata_path=cfg["metadata"],
        embedding_model_2=emb_model_2,
        embedding_tokenizer_2=emb_tokenizer_2,
        faiss_index_path_2=cfg.get("faiss_index_2"),
        metadata_path_2=cfg.get("metadata_2"),
    )
    print(f"✅ Pipeline ready.\n")
    return pipeline


# ── Output path helper ────────────────────────────────────────────────────────

def make_output_path(sentence: str, output_dir: str, index: int = 0) -> str:
    os.makedirs(output_dir, exist_ok=True)
    slug = sentence[:40].strip().replace(" ", "_")
    slug = "".join(c for c in slug if c.isalnum() or c in "_-")
    if not slug:
        slug = f"output_{index}"
    return os.path.join(output_dir, f"{slug}.mp4")


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_result(result: dict, verbose: bool, as_json: bool):
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\n{'─'*60}")
    print(f"📝  Input     : {result.get('input', '')}")
    print(f"✏️   Normalized: {result.get('normalized', '')}")

    if verbose:
        gloss = result.get("gloss", {})
        print(f"\n📖  Gloss structure:")
        print(json.dumps(gloss, ensure_ascii=False, indent=4))

        tokens = result.get("tokens", [])
        print(f"\n🔤  Tokens    : {' → '.join(tokens) if tokens else '(none)'}")

        retrievals = result.get("retrievals", {})
        top1 = {t: hits[0] if hits else None for t, hits in retrievals.items()}
        print(f"\n🔍  Top-1 retrievals:")
        print(json.dumps(top1, ensure_ascii=False, indent=4))

    print(f"{'─'*60}\n")


# ── Single translate call ─────────────────────────────────────────────────────

def translate_one(pipeline, sentence: str, output_path: str,
                  top_k: int, verbose: bool, as_json: bool) -> bool:
    """Returns True on success."""
    print(f"🔄  Translating: {sentence!r}")
    try:
        result = pipeline.run(sentence, output_path=output_path, top_k=top_k)
        if not result:
            print("❌  Pipeline returned no result.")
            return False
        result["output_video"] = output_path
        print_result(result, verbose, as_json)
        print(f"🎬  Video saved → {output_path}")
        return True
    except FileNotFoundError as e:
        print(f"❌  File not found: {e}")
    except Exception as e:
        print(f"❌  Error: {e}")
    return False


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(pipeline, args):
    batch_file = Path(args.batch)
    if not batch_file.exists():
        sys.exit(f"❌  Batch file not found: {batch_file}")

    sentences = [
        line.strip()
        for line in batch_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if not sentences:
        sys.exit("❌  Batch file is empty.")

    print(f"📂  Batch mode — {len(sentences)} sentence(s) from {batch_file}\n")
    success, fail = 0, 0

    for i, sentence in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}]", end=" ")
        out_path = make_output_path(sentence, args.output_dir, index=i)
        ok = translate_one(pipeline, sentence, out_path,
                           args.top_k, args.verbose, args.json)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\n✅  Done — {success} succeeded, {fail} failed.")


# ── Interactive REPL mode ─────────────────────────────────────────────────────

def run_interactive(pipeline, args):
    print("🤟  ViSL Interactive Translator  (type 'quit' or Ctrl-C to exit)\n")
    counter = 0
    while True:
        try:
            sentence = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not sentence:
            continue
        if sentence.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        counter += 1
        out_path = make_output_path(sentence, args.output_dir, index=counter)
        translate_one(pipeline, sentence, out_path,
                      args.top_k, args.verbose, args.json)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate API key
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        sys.exit(
            "❌  GOOGLE_API_KEY is not set.\n"
            "    Local: export GOOGLE_API_KEY='AIza...'  or add it to .env\n"
        )

    # Validate input mode
    if not args.sentence and not args.batch and not args.interactive:
        sys.exit(
            "❌  No input provided.\n"
            "    Pass a sentence, use --batch FILE, or use --interactive.\n"
            "    Run with --help for usage details."
        )

    # Build config & load models
    dialect_key, dialect_config, emb_path, emb_path_2 = build_dialect_config(args)
    emb_model, emb_tok, emb_model_2, emb_tok_2 = load_models(
        emb_path, emb_path_2, args.ensemble
    )

    # Build pipeline
    pipeline = build_pipeline(
        args, dialect_key, dialect_config,
        emb_model, emb_tok, emb_model_2, emb_tok_2,
    )

    # Run selected mode
    if args.batch:
        run_batch(pipeline, args)

    elif args.interactive:
        run_interactive(pipeline, args)

    else:
        # Single sentence
        out_path = args.output or make_output_path(args.sentence, args.output_dir)
        ok = translate_one(
            pipeline, args.sentence, out_path,
            args.top_k, args.verbose, args.json,
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()