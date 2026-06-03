# -*- coding: utf-8 -*-
"""
evaluate_latency.py
-------------------
Evaluate end-to-end and per-module latency of the ViSL pipeline over N sentences.

Usage (basic):
    python evaluate_latency.py --sentences data/eval_sentences.txt --n 50

Usage (custom pipeline args):
    python evaluate_latency.py \
        --sentences data/eval_sentences.txt \
        --n 100 \
        --model_type large \
        --dialect "🇳 North" \
        --output_dir results/ \
        --warmup 2
"""

import os
import json
import time
import logging
import argparse
import statistics
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepTiming:
    """Timing for a single pipeline step (in seconds)."""
    step2_normalize:     float = 0.0
    step3_gloss:         float = 0.0
    step4_segment:       float = 0.0
    step5_6_retrieve:    float = 0.0
    step7_pose:          float = 0.0
    end_to_end:          float = 0.0

    def to_ms(self) -> "StepTiming":
        """Return a copy with all values converted to milliseconds."""
        return StepTiming(**{k: v * 1000 for k, v in asdict(self).items()})


@dataclass
class SentenceResult:
    """Result for a single evaluated sentence."""
    sentence:  str
    dialect:   str
    success:   bool
    error:     Optional[str] = None
    timing:    Optional[StepTiming] = None
    gloss:     Optional[dict] = None
    tokens:    Optional[list] = None


@dataclass
class EvaluationReport:
    """Aggregated report over all N sentences."""
    n_total:        int = 0
    n_success:      int = 0
    n_failed:       int = 0
    dialect:        str = ""
    model_type:     str = ""
    results:        list[SentenceResult] = field(default_factory=list)
    stats:          dict = field(default_factory=dict)  # per-step stats


class TimedViSLPipeline:
    """
    Wraps ViSLPipeline and records wall-clock time for each step.
    Requires visl_pipeline.ViSLPipeline to be importable.
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def run_timed(self, input_text: str, output_path: str, top_k: int = 5) -> SentenceResult:
        timing = StepTiming()
        gloss, tokens = None, None

        t_total_start = time.perf_counter()

        # ── Step 2 ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            normalized = self._pipeline.step2_normalize(input_text)
        except Exception as e:
            return SentenceResult(
                sentence=input_text, dialect="", success=False, error=f"[step2] {e}"
            )
        timing.step2_normalize = time.perf_counter() - t0

        # ── Step 3 ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            gloss = self._pipeline.step3_gloss(normalized)
        except Exception as e:
            return SentenceResult(
                sentence=input_text, dialect="", success=False, error=f"[step3] {e}",
                timing=timing,
            )
        timing.step3_gloss = time.perf_counter() - t0

        if gloss is None:
            timing.end_to_end = time.perf_counter() - t_total_start
            return SentenceResult(
                sentence=input_text, dialect="", success=False,
                error="[step3] gloss returned None", timing=timing,
            )

        # ── Step 4 ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            tokens = self._pipeline.step4_segment(gloss)
        except Exception as e:
            return SentenceResult(
                sentence=input_text, dialect="", success=False, error=f"[step4] {e}",
                timing=timing, gloss=gloss,
            )
        timing.step4_segment = time.perf_counter() - t0

        # ── Step 5-6 ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            retrievals = self._pipeline.step5_6_retrieve(normalized, tokens, top_k=top_k)
        except Exception as e:
            return SentenceResult(
                sentence=input_text, dialect="", success=False, error=f"[step5-6] {e}",
                timing=timing, gloss=gloss, tokens=tokens,
            )
        timing.step5_6_retrieve = time.perf_counter() - t0

        # ── Step 7 ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            concat_pose = self._pipeline.step_7_skeleton_generation_and_pose_smoothing(retrievals)
            from pose_format.pose_visualizer import PoseVisualizer
            v = PoseVisualizer(concat_pose)
            v.save_video(output_path, v.draw())
        except Exception as e:
            timing.step7_pose = time.perf_counter() - t0
            timing.end_to_end = time.perf_counter() - t_total_start
            return SentenceResult(
                sentence=input_text, dialect="", success=False, error=f"[step7] {e}",
                timing=timing, gloss=gloss, tokens=tokens,
            )
        timing.step7_pose = time.perf_counter() - t0
        timing.end_to_end = time.perf_counter() - t_total_start

        return SentenceResult(
            sentence=input_text, dialect="", success=True,
            timing=timing, gloss=gloss, tokens=tokens,
        )


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

STEP_LABELS = {
    "step2_normalize":  "Step 2  — Text Normalize",
    "step3_gloss":      "Step 3  — Text → Gloss (Gemini)",
    "step4_segment":    "Step 4  — Word Segmentation",
    "step5_6_retrieve": "Step 5-6 — Embedding + Retrieval",
    "step7_pose":       "Step 7  — Pose Concat + Video",
    "end_to_end":       "END-TO-END",
}


def _compute_stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n":      len(values),
        "mean":   statistics.mean(values),
        "median": statistics.median(values),
        "stdev":  statistics.stdev(values) if len(values) > 1 else 0.0,
        "min":    min(values),
        "max":    max(values),
        "p90":    sorted(values)[int(len(values) * 0.90)],
        "p95":    sorted(values)[int(len(values) * 0.95)],
    }


def aggregate_results(results: list[SentenceResult]) -> dict:
    """Compute per-step stats (ms) from successful runs."""
    successful = [r for r in results if r.success and r.timing is not None]

    step_fields = list(asdict(StepTiming()).keys())
    all_timings: dict[str, list[float]] = {k: [] for k in step_fields}

    for r in successful:
        d = asdict(r.timing)
        for k in step_fields:
            all_timings[k].append(d[k] * 1000)   # convert to ms

    return {k: _compute_stats(v) for k, v in all_timings.items() if v}


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(report: EvaluationReport) -> None:
    sep = "═" * 72
    thin = "─" * 72

    print(f"\n{sep}")
    print(f"  ViSL LATENCY EVALUATION REPORT")
    print(f"  Dialect: {report.dialect}   |   Model: {report.model_type}")
    print(f"  Total: {report.n_total}   Success: {report.n_success}   Failed: {report.n_failed}")
    print(sep)

    header = f"{'Step':<42} {'Mean':>8} {'Median':>8} {'P90':>8} {'P95':>8} {'Max':>8}"
    print(header)
    print(thin)

    for field_key, label in STEP_LABELS.items():
        s = report.stats.get(field_key)
        if not s:
            continue
        is_e2e = (field_key == "end_to_end")
        prefix = "▶  " if is_e2e else "   "
        if is_e2e:
            print(thin)
        row = (
            f"{prefix}{label:<39} "
            f"{s['mean']:>7.1f}ms "
            f"{s['median']:>7.1f}ms "
            f"{s['p90']:>7.1f}ms "
            f"{s['p95']:>7.1f}ms "
            f"{s['max']:>7.1f}ms"
        )
        print(row)

    print(sep)

    if report.n_failed > 0:
        print(f"\n  ⚠  Failed sentences ({report.n_failed}):")
        for r in report.results:
            if not r.success:
                print(f"     • {r.sentence[:60]!r}  →  {r.error}")
        print()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(
    sentences: list[str],
    pipeline,                    # ViSLPipeline instance
    dialect: str,
    model_type: str,
    top_k: int = 5,
    warmup: int = 1,
    output_dir: str = "/tmp/visl_eval",
) -> EvaluationReport:
    """
    Run the pipeline over all sentences and return a full EvaluationReport.

    Args:
        sentences:   List of Vietnamese input sentences.
        pipeline:    An initialised ViSLPipeline instance.
        dialect:     Human-readable dialect label (for the report).
        model_type:  Model type label (for the report).
        top_k:       Number of FAISS neighbours per token.
        warmup:      Number of warm-up runs (not counted in stats).
        output_dir:  Directory for temporary video outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    timed = TimedViSLPipeline(pipeline)

    # ── Warm-up ────────────────────────────────────────────────────────────
    if warmup > 0 and sentences:
        logger.info(f"Running {warmup} warm-up sentence(s)...")
        for i in range(min(warmup, len(sentences))):
            out = tempfile.NamedTemporaryFile(suffix=".mp4", dir=output_dir, delete=False).name
            timed.run_timed(sentences[i], out, top_k=top_k)
        logger.info("Warm-up done.\n")

    # ── Main evaluation loop ───────────────────────────────────────────────
    all_results: list[SentenceResult] = []

    for i, sentence in enumerate(sentences):
        logger.info(f"[{i+1:>4}/{len(sentences)}] {sentence[:70]!r}")
        out = tempfile.NamedTemporaryFile(suffix=".mp4", dir=output_dir, delete=False).name
        result = timed.run_timed(sentence, out, top_k=top_k)
        result.dialect = dialect

        if result.success:
            t = result.timing
            logger.info(
                f"         ✓  e2e={t.end_to_end*1000:.0f}ms  "
                f"gloss={t.step3_gloss*1000:.0f}ms  "
                f"retrieval={t.step5_6_retrieve*1000:.0f}ms  "
                f"pose={t.step7_pose*1000:.0f}ms"
            )
        else:
            logger.warning(f"         ✗  {result.error}")

        all_results.append(result)

    # ── Build report ───────────────────────────────────────────────────────
    n_success = sum(1 for r in all_results if r.success)
    report = EvaluationReport(
        n_total=len(all_results),
        n_success=n_success,
        n_failed=len(all_results) - n_success,
        dialect=dialect,
        model_type=model_type,
        results=all_results,
        stats=aggregate_results(all_results),
    )
    return report


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_report_json(report: EvaluationReport, path: str) -> None:
    """Serialise report to JSON (timings in ms)."""
    data = {
        "n_total":    report.n_total,
        "n_success":  report.n_success,
        "n_failed":   report.n_failed,
        "dialect":    report.dialect,
        "model_type": report.model_type,
        "stats_ms":   report.stats,
        "results": [
            {
                "sentence": r.sentence,
                "dialect":  r.dialect,
                "success":  r.success,
                "error":    r.error,
                "timing_ms": {
                    k: v * 1000 for k, v in asdict(r.timing).items()
                } if r.timing else None,
                "gloss":   r.gloss,
                "tokens":  r.tokens,
            }
            for r in report.results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved → {path}")


def save_report_csv(report: EvaluationReport, path: str) -> None:
    """Save per-sentence timings as CSV for further analysis."""
    import csv
    step_fields = list(asdict(StepTiming()).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "dialect", "success", "error"] +
                        [f"{k}_ms" for k in step_fields])
        for r in report.results:
            timing_vals = (
                [asdict(r.timing)[k] * 1000 for k in step_fields]
                if r.timing else [""] * len(step_fields)
            )
            writer.writerow([r.sentence, r.dialect, r.success, r.error or ""] + timing_vals)
    logger.info(f"CSV saved    → {path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate ViSL pipeline latency over N sentences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sentences",   required=True,
                   help="Path to .txt file with one Vietnamese sentence per line")
    p.add_argument("--n", type=int, default=None,
                   help="Max number of sentences to evaluate (default: all)")
    p.add_argument("--dialect", default="🇳 North",
                   choices=["🇳 North", "🇨 Central", "🇸 South"],
                   help="Regional dialect to evaluate")
    p.add_argument("--model_type",  default="large",
                   choices=["base", "large"],
                   help="Embedding model variant")
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of FAISS neighbours per token")
    p.add_argument("--warmup", type=int, default=2,
                   help="Number of warm-up runs before measuring")
    p.add_argument("--output_dir",  default="eval_outputs/",
                   help="Directory for temp video files")
    p.add_argument("--save_json", default=None,
                   help="Path to save JSON report (e.g. results/report.json)")
    p.add_argument("--save_csv", default=None,
                   help="Path to save CSV report (e.g. results/report.csv)")
    p.add_argument("--ensemble", action="store_true",
                   help="Use ensemble mode (base + large)")
    p.add_argument("--poses_path", default="./poses/",
                   help="Path to pose files directory")
    return p


def main():
    args = build_arg_parser().parse_args()

    # ── Load sentences ─────────────────────────────────────────────────────
    sentences_path = Path(args.sentences)
    if not sentences_path.exists():
        raise FileNotFoundError(f"Sentence file not found: {sentences_path}")

    with open(sentences_path, encoding="utf-8") as f:
        all_sentences = [line.strip() for line in f if line.strip()]

    if args.n is not None:
        all_sentences = all_sentences[: args.n]

    logger.info(f"Loaded {len(all_sentences)} sentences from {sentences_path}")
    logger.info(f"Dialect: {args.dialect}  |  Model: {args.model_type}  |  Ensemble: {args.ensemble}")

    # ── Build pipeline ─────────────────────────────────────────────────────
    from dotenv import load_dotenv
    load_dotenv()

    from transformers import AutoModel, AutoTokenizer
    from visl_pipeline import ViSLPipeline

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {_device}")

    def _load(path, label):
        logger.info(f"Loading {label} from {path} ...")
        model = AutoModel.from_pretrained(
            path, trust_remote_code=True, ignore_mismatched_sizes=True
        ).to(_device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        logger.info(f"  ✓ {label} loaded")
        return model, tokenizer

    mt = args.model_type
    emb_model, emb_tokenizer = _load(f"tkhangg0910/viconbert-{mt}", "Model-1")

    emb_model_2 = emb_tokenizer_2 = None
    if args.ensemble:
        mt2 = "base" if mt == "large" else "large"
        emb_model_2, emb_tokenizer_2 = _load(f"tkhangg0910/viconbert-{mt2}", "Model-2")

    DIALECT_DB = {
        "🇳 North":   ("AB", "AB"),
        "🇨 Central": ("AT", "AT"),
        "🇸 South":   ("AN", "AN"),
    }
    db_code, _ = DIALECT_DB[args.dialect]

    pipeline = ViSLPipeline(
        poses_path=args.poses_path,
        embedding_model=emb_model,
        embedding_tokenizer=emb_tokenizer,
        faiss_index_path=f"./pose_databases/index_mean_{db_code}_{mt}.faiss",
        metadata_path=f"./pose_databases/metadata_mean_{db_code}_{mt}.json",
        embedding_model_2=emb_model_2,
        embedding_tokenizer_2=emb_tokenizer_2,
        faiss_index_path_2=(
            f"./pose_databases/index_mean_{db_code}_{'base' if mt=='large' else 'large'}.faiss"
            if args.ensemble else None
        ),
        metadata_path_2=(
            f"./pose_databases/metadata_mean_{db_code}_{'base' if mt=='large' else 'large'}.json"
            if args.ensemble else None
        ),
    )

    # ── Run evaluation ─────────────────────────────────────────────────────
    report = evaluate(
        sentences=all_sentences,
        pipeline=pipeline,
        dialect=args.dialect,
        model_type=args.model_type + (" (ensemble)" if args.ensemble else ""),
        top_k=args.top_k,
        warmup=args.warmup,
        output_dir=args.output_dir,
    )

    # ── Print ──────────────────────────────────────────────────────────────
    print_report(report)

    # ── Save ───────────────────────────────────────────────────────────────
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        save_report_json(report, args.save_json)

    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        save_report_csv(report, args.save_csv)


if __name__ == "__main__":
    main()