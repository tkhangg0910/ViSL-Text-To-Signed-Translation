import os
import json
import tempfile
import threading
import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
from visl_pipeline import ViSLPipeline

load_dotenv()
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ViSL Translator")

    # ── run mode ──────────────────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--advanced", action="store_true",
        help="Use advanced configuration (custom paths for each component)",
    )
    mode_group.add_argument(
        "--ensemble", action="store_true",
        help="Ensemble mode: run both base and large models in parallel, "
             "merge and sort results by similarity score",
    )

    # ── basic / ensemble shared arg ───────────────────────────────────────────
    parser.add_argument("--model_type", type=str, default="large",
                        help="Model type used in single mode: base / large (default: large)")

    # ── advanced args ─────────────────────────────────────────────────────────
    parser.add_argument("--poses_path",      type=str)
    parser.add_argument("--embedding_model", type=str)

    parser.add_argument("--index_north",  type=str)
    parser.add_argument("--meta_north",   type=str)

    parser.add_argument("--index_central", type=str)
    parser.add_argument("--meta_central",  type=str)

    parser.add_argument("--index_south",  type=str)
    parser.add_argument("--meta_south",   type=str)

    # ── ensemble extra args ───────────────────────────────────────────────────
    # Model-2 paths (optional; defaults built automatically from model_type_2)
    parser.add_argument("--model_type_2", type=str, default=None,
                        help="Second model type for ensemble (default: the other of base/large)")

    parser.add_argument("--index_north_2",   type=str)
    parser.add_argument("--meta_north_2",    type=str)
    parser.add_argument("--index_central_2", type=str)
    parser.add_argument("--meta_central_2",  type=str)
    parser.add_argument("--index_south_2",   type=str)
    parser.add_argument("--meta_south_2",    type=str)

    return parser.parse_args()


args = parse_args()

# ── Resolve paths ─────────────────────────────────────────────────────────────

POSES_PATH = "./poses/"

if args.advanced:
    # ── ADVANCED MODE ──────────────────────────────────────────────────────────
    POSES_PATH              = args.poses_path
    EMBEDDING_MODEL_PATH    = args.embedding_model
    EMBEDDING_MODEL_PATH_2  = None   # not supported in advanced mode

    DIALECT_CONFIG = {
        "🇳 North": {
            "faiss_index": args.index_north,  "metadata": args.meta_north,
            "faiss_index_2": None,            "metadata_2": None,
        },
        "🇨 Central": {
            "faiss_index": args.index_central, "metadata": args.meta_central,
            "faiss_index_2": None,             "metadata_2": None,
        },
        "🇸 South": {
            "faiss_index": args.index_south,  "metadata": args.meta_south,
            "faiss_index_2": None,            "metadata_2": None,
        },
    }

elif args.ensemble:
    # ── ENSEMBLE MODE ──────────────────────────────────────────────────────────
    # Primary model: --model_type (default "large")
    # Secondary model: --model_type_2 (auto-infer as the other of base/large)
    mt1 = args.model_type
    mt2 = args.model_type_2 or ("base" if mt1 == "large" else "large")

    EMBEDDING_MODEL_PATH   = f"tkhangg0910/viconbert-{mt1}"
    EMBEDDING_MODEL_PATH_2 = f"tkhangg0910/viconbert-{mt2}"

    def _db(dialect_code, model_type):
        return {
            "🇳 North":   f"./pose_databases/index_mean_AB_{model_type}.faiss",
            "🇨 Central": f"./pose_databases/index_mean_AT_{model_type}.faiss",
            "🇸 South":   f"./pose_databases/index_mean_AN_{model_type}.faiss",
        }[dialect_code], {
            "🇳 North":   f"./pose_databases/metadata_mean_AB_{model_type}.json",
            "🇨 Central": f"./pose_databases/metadata_mean_AT_{model_type}.json",
            "🇸 South":   f"./pose_databases/metadata_mean_AN_{model_type}.json",
        }[dialect_code]

    DIALECT_CONFIG = {}
    for dialect in ["🇳 North", "🇨 Central", "🇸 South"]:
        idx1, meta1 = _db(dialect, mt1)
        idx2, meta2 = _db(dialect, mt2)

        # Allow CLI overrides for dialect-specific ensemble paths
        if dialect == "🇳 North":
            idx2  = args.index_north_2  or idx2
            meta2 = args.meta_north_2   or meta2
        elif dialect == "🇨 Central":
            idx2  = args.index_central_2 or idx2
            meta2 = args.meta_central_2  or meta2
        elif dialect == "🇸 South":
            idx2  = args.index_south_2  or idx2
            meta2 = args.meta_south_2   or meta2

        DIALECT_CONFIG[dialect] = {
            "faiss_index":   idx1,  "metadata":   meta1,
            "faiss_index_2": idx2,  "metadata_2": meta2,
        }

    print(f"[Ensemble] Model-1: {EMBEDDING_MODEL_PATH}")
    print(f"[Ensemble] Model-2: {EMBEDDING_MODEL_PATH_2}")

else:
    # ── BASIC MODE ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL_PATH   = f"tkhangg0910/viconbert-{args.model_type}"
    EMBEDDING_MODEL_PATH_2 = None

    DIALECT_CONFIG = {
        "🇳 North": {
            "faiss_index": f"./pose_databases/index_mean_AB_{args.model_type}.faiss",
            "metadata":    f"./pose_databases/metadata_mean_AB_{args.model_type}.json",
            "faiss_index_2": None, "metadata_2": None,
        },
        "🇨 Central": {
            "faiss_index": f"./pose_databases/index_mean_AT_{args.model_type}.faiss",
            "metadata":    f"./pose_databases/metadata_mean_AT_{args.model_type}.json",
            "faiss_index_2": None, "metadata_2": None,
        },
        "🇸 South": {
            "faiss_index": f"./pose_databases/index_mean_AN_{args.model_type}.faiss",
            "metadata":    f"./pose_databases/metadata_mean_AN_{args.model_type}.json",
            "faiss_index_2": None, "metadata_2": None,
        },
    }

# ── Load embedding model(s) — runs once at startup ────────────────────────────

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_path: str, label: str):
    print(f"⏳ Loading {label}: {model_path} ...")
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, ignore_mismatched_sizes=True
    ).to(_device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    print(f"✅ {label} loaded on {_device}")
    return model, tokenizer


if args.ensemble:
    # Load both models in parallel threads to speed up startup
    _m1, _t1, _m2, _t2 = None, None, None, None

    def _load1():
        global _m1, _t1
        _m1, _t1 = _load_model(EMBEDDING_MODEL_PATH, "Model-1")

    def _load2():
        global _m2, _t2
        _m2, _t2 = _load_model(EMBEDDING_MODEL_PATH_2, "Model-2")

    _th1 = threading.Thread(target=_load1, daemon=True)
    _th2 = threading.Thread(target=_load2, daemon=True)
    _th1.start(); _th2.start()
    _th1.join();  _th2.join()

    _emb_model,       _emb_tokenizer       = _m1, _t1
    _emb_model_2,     _emb_tokenizer_2     = _m2, _t2
else:
    _emb_model, _emb_tokenizer = _load_model(EMBEDDING_MODEL_PATH, "Embedding model")
    _emb_model_2 = _emb_tokenizer_2 = None

# ── Pipeline cache — lazy-init per dialect, reused across requests ────────────

_pipeline_cache: dict[str, "ViSLPipeline"] = {}
_cache_lock = threading.Lock()


def get_pipeline(dialect_key: str) -> "ViSLPipeline":
    with _cache_lock:
        if dialect_key not in _pipeline_cache:
            cfg = DIALECT_CONFIG[dialect_key]
            print(f"⏳ Initializing pipeline for: {dialect_key}")
            _pipeline_cache[dialect_key] = ViSLPipeline(
                poses_path=POSES_PATH,
                # model 1
                embedding_model=_emb_model,
                embedding_tokenizer=_emb_tokenizer,
                faiss_index_path=cfg["faiss_index"],
                metadata_path=cfg["metadata"],
                # model 2 (None in basic/advanced mode → ignored inside pipeline)
                embedding_model_2=_emb_model_2,
                embedding_tokenizer_2=_emb_tokenizer_2,
                faiss_index_path_2=cfg.get("faiss_index_2"),
                metadata_path_2=cfg.get("metadata_2"),
            )
            print(f"✅ Pipeline ready: {dialect_key}")
        return _pipeline_cache[dialect_key]


# ── Core translate function ───────────────────────────────────────────────────

def translate(input_text: str, dialect: str):
    """Run the full ViSL pipeline and return video + intermediate outputs."""

    if not input_text.strip():
        return None, "⚠️ Please enter a sentence to translate.", "", ""

    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        return None, "⚠️ GOOGLE_API_KEY is not set. Please configure it before running.", "", ""

    try:
        pipeline     = get_pipeline(dialect)
        os.makedirs("./outputs", exist_ok=True)
        output_video = tempfile.NamedTemporaryFile(suffix=".mp4",dir="./outputs", delete=False).name
        result       = pipeline.run(input_text, output_path=output_video, top_k=5)

        if not result:
            return None, "❌ Pipeline returned no result.", "", ""

        gloss_str      = json.dumps(result.get("gloss", {}),      ensure_ascii=False, indent=2)
        tokens         = result.get("tokens", [])
        tokens_str     = "  →  ".join(tokens) if tokens else "(no tokens)"
        retrievals     = result.get("retrievals", {})
        top1           = {t: hits[0] if hits else None for t, hits in retrievals.items()}
        retrievals_str = json.dumps(top1, ensure_ascii=False, indent=2)

        return output_video, gloss_str, tokens_str, retrievals_str

    except FileNotFoundError as e:
        return None, f"❌ File not found: {e}", "", ""
    except Exception as e:
        return None, f"❌ Error: {e}", "", ""


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#title  { text-align: center; margin-bottom: 4px; }
#banner { text-align: center; color: #666; margin-bottom: 16px; font-size: 15px; }
#mode-badge { text-align: center; margin-bottom: 8px; }
footer  { display: none !important; }
"""

EXAMPLES = [
    ["Ngày mai tôi đến ngân hàng."],
    ["Hôm nay trời nắng đẹp."],
    ["Bạn tên gì?"],
    ["Tôi muốn học ngôn ngữ ký hiệu."],
]

# Build a small badge showing the active mode
if args.ensemble:
    _mode_label = "⚡ **Ensemble mode** — base + large models, parallel inference + merged results"
elif args.advanced:
    _mode_label = "⚙️ **Advanced mode** — custom paths"
else:
    _mode_label = f"🔧 **Basic mode** — `{args.model_type}` model"

with gr.Blocks(css=CSS, title="ViSL Translator", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🤟 ViSL — Vietnamese Sign Language Translator", elem_id="title")
    gr.Markdown(_mode_label, elem_id="mode-badge")
    gr.Markdown(
        "Enter a Vietnamese sentence, select a regional dialect, and watch the sign language output.",
        elem_id="banner",
    )

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### ✍️ Input")

            dialect_radio = gr.Radio(
                choices=list(DIALECT_CONFIG.keys()),
                value=list(DIALECT_CONFIG.keys())[0],
                label="Regional Dialect",
                info="Each dialect uses a separate sign language vocabulary database.",
            )

            text_input = gr.Textbox(
                label="Vietnamese Sentence",
                placeholder="e.g. Ngày mai tôi đến phòng giao dịch ngân hàng.",
                lines=4,
            )

            translate_btn = gr.Button("▶  Translate", variant="primary", size="lg")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input],
                label="Example Sentences",
            )

        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 🎬 Sign Language Output")

            video_output = gr.Video(
                label="Generated Pose Video",
                autoplay=True,
                # show_download_button=True,
            )

    with gr.Accordion("🔍 Pipeline Details (Gloss / Tokens / Retrieval)", open=False):
        gr.Markdown(
            "Intermediate outputs from each step of the translation pipeline.",
            elem_id="banner",
        )
        with gr.Row():
            gloss_output = gr.Code(
                label="Step 3 — Gloss Structure (JSON)",
                language="json",
                lines=12,
            )
            retrievals_output = gr.Code(
                label="Step 5–6 — Top-1 Retrieval per Token (JSON)",
                language="json",
                lines=12,
            )
        tokens_output = gr.Textbox(
            label="Step 4 — Word Segmentation Tokens",
            lines=2,
            interactive=False,
            placeholder="Tokens will appear here after translation...",
        )

    translate_btn.click(
        fn=translate,
        inputs=[text_input, dialect_radio],
        outputs=[video_output, gloss_output, tokens_output, retrievals_output],
        show_progress="full",
    )


if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set.\n"
            "  Local : export GOOGLE_API_KEY='AIza...'  or add to .env file\n"
            "  Colab : use Colab Secrets (left sidebar) and add GOOGLE_API_KEY"
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )