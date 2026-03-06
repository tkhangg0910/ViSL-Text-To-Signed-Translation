
import os
import json
import tempfile
import threading
import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
from visl_pipeline import ViSLPipeline
# Load environment variables
load_dotenv()
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ViSL Translator")

    # mode
    parser.add_argument("--advanced", action="store_true",
                        help="Use advanced configuration")

    # basic arg
    parser.add_argument("--model_type", type=str, default="large",
                        help="Model type: base / large")

    # advanced args
    parser.add_argument("--poses_path", type=str)
    parser.add_argument("--embedding_model", type=str)

    parser.add_argument("--index_north", type=str)
    parser.add_argument("--meta_north", type=str)

    parser.add_argument("--index_central", type=str)
    parser.add_argument("--meta_central", type=str)

    parser.add_argument("--index_south", type=str)
    parser.add_argument("--meta_south", type=str)

    return parser.parse_args()
args = parse_args()

if not args.advanced:
    # BASIC MODE
    POSES_PATH = "./poses/"

    EMBEDDING_MODEL_PATH = f"tkhangg0910/viconbert-{args.model_type}"

    DIALECT_CONFIG = {
        "🇳 North": {
            "faiss_index": f"./pose_databases/index_mean_AB_{args.model_type}.faiss",
            "metadata": f"./pose_databases/metadata_mean_AB_{args.model_type}.json",
        },
        "🇨 Central": {
            "faiss_index": f"./pose_databases/index_mean_AT_{args.model_type}.faiss",
            "metadata": f"./pose_databases/metadata_mean_AT_{args.model_type}.json",
        },
        "🇸 South": {
            "faiss_index": f"./pose_databases/index_mean_AN_{args.model_type}.faiss",
            "metadata": f"./pose_databases/metadata_mean_AN_{args.model_type}.json",
        },
    }

else:
    # ADVANCED MODE
    POSES_PATH = args.poses_path
    EMBEDDING_MODEL_PATH = args.embedding_model

    DIALECT_CONFIG = {
        "🇳 North": {
            "faiss_index": args.index_north,
            "metadata": args.meta_north,
        },
        "🇨 Central": {
            "faiss_index": args.index_central,
            "metadata": args.meta_central,
        },
        "🇸 South": {
            "faiss_index": args.index_south,
            "metadata": args.meta_south,
        },
    }

# LOAD EMBEDDING MODEL — runs once at startup
print("⏳ Loading embedding model...")
_device        = "cuda" if torch.cuda.is_available() else "cpu"
_emb_model     = AutoModel.from_pretrained(
    EMBEDDING_MODEL_PATH, trust_remote_code=True, ignore_mismatched_sizes=True
).to(_device).eval()
_emb_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH, use_fast=True)
print(f"Embedding model loaded on {_device}")

# PIPELINE CACHE — lazy-init per dialect, reused across requests
_pipeline_cache: dict[str, "ViSLPipeline"] = {}
_cache_lock = threading.Lock()

def get_pipeline(dialect_key: str) -> "ViSLPipeline":
    with _cache_lock:
        if dialect_key not in _pipeline_cache:
            cfg = DIALECT_CONFIG[dialect_key]
            print(f"⏳ Initializing pipeline for: {dialect_key}")
            _pipeline_cache[dialect_key] = ViSLPipeline(
                poses_path          = POSES_PATH,
                embedding_model     = _emb_model,
                embedding_tokenizer = _emb_tokenizer,
                faiss_index_path    = cfg["faiss_index"],
                metadata_path       = cfg["metadata"],
            )
            print(f"✅ Pipeline ready: {dialect_key}")
        return _pipeline_cache[dialect_key]


# CORE TRANSLATE FUNCTION
def translate(input_text: str, dialect: str):
    """Run the full ViSL pipeline and return video + intermediate outputs."""

    if not input_text.strip():
        return None, "⚠️ Please enter a sentence to translate.", "", ""

    # Verify API key is available in environment
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        return None, "⚠️ GOOGLE_API_KEY is not set. Please configure it before running.", "", ""

    try:
        pipeline     = get_pipeline(dialect)
        output_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        result       = pipeline.run(input_text, output_path=output_video, top_k=5)

        if not result:
            return None, "❌ Pipeline returned no result.", "", ""

        normalized     = result.get("normalized", [])
        print(normalized)
        # Gloss JSON from Step 3
        gloss_str = json.dumps(result.get("gloss", {}), ensure_ascii=False, indent=2)

        # Token chain from Step 4
        tokens     = result.get("tokens", [])
        tokens_str = "  →  ".join(tokens) if tokens else "(no tokens)"

        # Top-1 retrieval per token from Step 5-6
        retrievals     = result.get("retrievals", {})
        top1           = {t: hits[0] if hits else None for t, hits in retrievals.items()}
        retrievals_str = json.dumps(top1, ensure_ascii=False, indent=2)

        return output_video, gloss_str, tokens_str, retrievals_str

    except FileNotFoundError as e:
        return None, f"❌ File not found: {e}", "", ""
    except Exception as e:
        return None, f"❌ Error: {e}", "", ""


# UI
CSS = """
#title  { text-align: center; margin-bottom: 4px; }
#banner { text-align: center; color: #666; margin-bottom: 16px; font-size: 15px; }
footer  { display: none !important; }
"""

EXAMPLES = [
    ["Ngày mai tôi đến ngân hàng."],
    ["Hôm nay trời nắng đẹp."],
    ["Bạn tên gì?"],
    ["Tôi muốn học ngôn ngữ ký hiệu."],
]

with gr.Blocks(css=CSS, title="ViSL Translator", theme=gr.themes.Soft()) as demo:

    # Header
    gr.Markdown("# 🤟 ViSL — Vietnamese Sign Language Translator", elem_id="title")
    gr.Markdown(
        "Enter a Vietnamese sentence, select a regional dialect, and watch the sign language output.",
        elem_id="banner",
    )

    with gr.Row(equal_height=False):

        # Left: Input panel 
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

        # Right: Output panel 
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 🎬 Sign Language Output")

            video_output = gr.Video(
                label="Generated Pose Video",
                autoplay=True,
                show_download_button=True,
            )

    # Intermediate steps (collapsed by default) 
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

    # Wire up translate button
    translate_btn.click(
        fn=translate,
        inputs=[text_input, dialect_radio],
        outputs=[video_output, gloss_output, tokens_output, retrievals_output],
        show_progress="full",
    )


if __name__ == "__main__":
    # Verify API key exists before launching
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