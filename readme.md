# 🤟 ViSL — Vietnamese Sign Language Text-to-Signed Translation

Convert Vietnamese spoken text into Vietnamese Sign Language (ViSL) pose videos using a multi-step NLP + retrieval pipeline.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Pipeline Architecture](#pipeline-architecture)

---

## Overview

This project translates Vietnamese sentences into sign language skeleton animations through a 7-step pipeline:

1. **Text normalization** — spelling correction & punctuation
2. **Text-to-Gloss** — convert sentence to ViSL gloss structure via Gemini AI
3. **Word segmentation** — tokenize gloss tokens into sign units
4. **Contextual embedding** — embed each token using a fine-tuned PhoBERT model
5. **Vector retrieval** — look up the closest sign pose from a FAISS database
6. **Pose concatenation & smoothing** — stitch poses into a continuous animation
7. **Video rendering** — output an `.mp4` video

Three regional dialect databases are supported: **North**, **Central**, and **South**.

---

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU (recommended) or CPU
- A valid [Google Gemini API key](https://aistudio.google.com/app/apikey)
- Node.js (optional, only if generating `.docx` docs)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tkhangg0910/ViSL-Text-To-Signed-Translation.git
cd ViSL-Text-To-Signed-Translation
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install spoken-to-signed-translation

```bash
git clone https://github.com/ZurichNLP/spoken-to-signed-translation
cd spoken-to-signed-translation && pip install .
cd ..
```

### 4. Install custom transformers (PhoBERT / BARTpho support)

```bash
git clone --single-branch \
    --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet \
    https://github.com/datquocnguyen/transformers.git

pip3 install -e transformers
```

### 5. Download FAISS vector databases

```bash
gdown "https://drive.google.com/drive/folders/1_I3tyZ4u6DrXCGNHeLmGerwYqkemjP6F" \
    --folder -O pose_databases
```

This downloads three dialect databases:

| File | Dialect |
|------|---------|
| `index_mean_AB.faiss` + `metadata_mean_AB.json` | 🇳 North |
| `index_mean_AT.faiss` + `metadata_mean_AT.json` | 🇨 Central |
| `index_mean_AN.faiss` + `metadata_mean_AN.json` | 🇸 South |

### 6. Download pose library

```python
import gdown

url = "https://drive.google.com/uc?id=1b5FIh31obzF6dbex4s9C68s2tcBAhsaq"
gdown.download(url, "poses.zip", quiet=False)
```

Then unzip:

```bash
unzip poses.zip
```

After this step your directory should look like:

```
ViSL-Text-To-Signed-Translation/
├── app.py
├── pipeline.py
├── requirements.txt
├── poses/                     ← extracted pose files
└── pose_databases/
    ├── index_mean_AB.faiss
    ├── metadata_mean_AB.json
    ├── index_mean_AT.faiss
    ├── metadata_mean_AT.json
    ├── index_mean_AN.faiss
    └── metadata_mean_AN.json
```

---

## Configuration

### Set your Google Gemini API key

**Local:**

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=AIza...
```

**Google Colab:**

Go to the left sidebar → 🔑 **Secrets**, add a secret named `GOOGLE_API_KEY`, then in your first notebook cell:

```python
import os
from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
```

### Update file paths in `app.py`

Open `app.py` and verify these paths match your environment:

```python
DIALECT_CONFIG = {
    "🇳 North": {
        "faiss_index": "/content/index_mean_AB.faiss",   # ← update if needed
        "metadata":    "/content/metadata_mean_AB.json",
    },
    ...
}

POSES_PATH = "/content/poses/"   # ← update if needed
```

---

## Running the App

### Local

```bash
python app.py
```

Open your browser at `http://localhost:7860`.

### Google Colab

```python
!python app.py
```

Gradio will print a public `*.gradio.live` URL — open it in any browser.

> **Note:** `share=True` is enabled by default so the app is accessible outside Colab.

---

## Pipeline Architecture

```
Input Text (Vietnamese)
        │
        ▼
 Step 1-2: Normalize Text
        │
        ▼
 Step 3: Text → Gloss (Gemini AI)
        │   { "S": ["tôi"], "V": ["ăn"], ... }
        ▼
 Step 4: Word Segmentation
        │   ["tôi", "ăn", "táo"]
        ▼
 Step 5-6: Contextual Embedding + FAISS Retrieval
        │   each token → nearest pose(s) in vector DB
        ▼
 Step 7: Pose Concatenation + Smoothing
        │
        ▼
   Output Video (.mp4)
```

| Step | Model / Tool |
|------|-------------|
| Normalization | `underthesea` |
| Text-to-Gloss | `gemini-2.5-flash` |
| Word Segmentation | `tkhangg0910/viVSL-word-segmentation` |
| Embedding | `tkhangg0910/viconbert-large` |
| Retrieval | FAISS + dialect-specific metadata |
| Pose rendering | `pose-format` + `PoseVisualizer` |

---

## License

This project is for research and educational purposes. See [LICENSE](LICENSE) for details.