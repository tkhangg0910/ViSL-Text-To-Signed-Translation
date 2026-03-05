import os
import re
import json
import torch
import faiss
import numpy as np
from string import Template
from functools import lru_cache
from dotenv import load_dotenv
from transformers import AutoModel, PhobertTokenizerFast,AutoTokenizer
import numpy as np
from pose_format import Pose
from pose_format.utils.generic import (
    correct_wrists,
    normalize_pose_size,
    pose_normalization_info,
    reduce_holistic,
)
from spoken_to_signed.gloss_to_pose.concatenate import concatenate_poses
from typing import Union
from pose_anonymization.appearance import (
    remove_appearance,
    transfer_appearance,
)
from underthesea import text_normalize as text_normalize_uts
from text_normalizer import text_normalize
from span_extractor import SpanExtractor
def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: sửa lỗi chính tả, chuẩn hóa dấu câu và format."""
    return text_normalize_uts(text)


from google import genai

GLOSS_PROMPT = """
You are a helpful assistant who helps glossify sentences in ViSL (Vietnamese Sign Language) into sign language glosses.
Your task is to convert spoken language text into glossed sign language sentences following specific formatting rules.

**Step 1**: Apply **Rule 1** - Remove Unnecessary Words:
Remove the following types of words from the sentence:

- Determiners: mỗi, từng, mọi, cái, các, những, mấy
- Aspect markers / tense markers: đã, sẽ, đang, vừa, mới, từng, xong, rồi
- Modal particles / filler words: à, ạ, ấy, chứ, nhé, nhỉ, vậy, đâu, chắc, chăng, hả, hử,...
- Exclamations / emphasis words: ơi, vâng, dạ, ôi, trời ơi, eo ôi, kìa,...
- Emphasis or focus particles: cả, chính, đích, đúng, chỉ, những, đến, tận, ngay,...
- Modal verbs: nên, cần, phải, có thể, bị, được, mong, chúc, ước, muốn, dám, định, đành...
- Demonstrative pronoun(Noun + DP): này, kia, nọ...
Example:
"Tôi đã ăn hết cả ba quả táo rồi." → "Tôi táo ba ăn"

**Step 2**: Using the result from Step 1, apply the remaining rules below:

- **Rule 2**: Change order of noun and counting words → from "Count + Noun" to "Noun + Count"
- **Rule 3**: Change the order of verbs and negative words → from "Verb + Negative" to "Negative + Verb"
- **Rule 4**: Change the order of complements and verbs → "Subject + Complement + Verb"
  Extended format: Time (optional) + Subject + Complement + Quantity (optional) + Place (optional) + Verb
- **Rule 5**: Change word order in interrogative sentences

**Output**: JSON with glossed structure. Example:
Hôm qua tôi ăn bánh mì cay Hàn Quốc ở Hà Nội →
{
  "TIME": ["hôm qua"],
  "S": ["tôi"],
  "O": ["bánh mì", "cay", "Hàn Quốc"],
  "PLACE": ["Hà Nội"],
  "V": ["ăn"]
}

Note: Only output the final JSON without any explanation or gloss sentence.

Sentence:
${sentence}
"""

_prompt_template = Template(GLOSS_PROMPT)


def parse_markdown_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            print(f"[Gloss] JSON decode error: {e}")
    else:
        print("[Gloss] Không tìm thấy nội dung JSON.")
    return None


@lru_cache(maxsize=1)
def _get_gemini_client():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


def text_to_gloss(text: str) -> dict | None:
    """Chuyển câu tiếng Việt sang cấu trúc gloss dạng JSON."""
    client = _get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=_prompt_template.substitute(sentence=text),
    )
    return parse_markdown_json(response.text)


def gloss_to_token_list(gloss: dict) -> list[str]:
    """Flatten gloss dict theo thứ tự TIME → S → O → PLACE → V thành list token."""
    order = ["TIME", "S", "O", "Q", "PLACE", "V"]
    tokens = []
    for key in order:
        if key in gloss:
            tokens.extend(gloss[key])
    for key, values in gloss.items():
        if key not in order:
            tokens.extend(values)
    return tokens

def read_pose(pose_path: str):
    pose_path = os.path.join(pose_path)
    with open(pose_path, "rb") as f:
        return Pose.read(f.read())

def gloss_to_pose(retrieved_glosses_path, anonymize: Union[bool, Pose] = False,) -> Pose:
    poses = [read_pose(path) for path in retrieved_glosses_path]
    # Anonymize poses
    if anonymize:
        if isinstance(anonymize, Pose):
            print("Transferring appearance...")
            poses = [transfer_appearance(pose, anonymize) for pose in poses]
        else:
            print("Removing appearance...")
            poses = [remove_appearance(pose) for pose in poses]

    # Concatenate the poses to create a single pose
    return concatenate_poses(poses)

from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForTokenClassification


class WordSegmenter:
    MODEL_PATH = "tkhangg0910/viVSL-word-segmentation"

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(self.MODEL_PATH)
        self.nlp = hf_pipeline(
            "token-classification", model=model, tokenizer=tokenizer, device=device
        )

    def segment(self, text: str) -> list[str]:
        """Tách văn bản thành danh sách từ/cụm từ."""
        ner_results = self.nlp(text)
        merged = ""
        for e in ner_results:
            word = e["word"]
            if "##" in word:
                merged += word.replace("##", "")
            elif e["entity"] == "LABEL_1":  
                merged += "_" + word
            else:  
                merged += " " + word
        return [s.replace("_", " ") for s in merged.split()]


class EmbeddingRetriever:
    def __init__(
        self,
        embedding_model,     
        tokenizer,             
        faiss_index_path: str,
        metadata_path: str,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = embedding_model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.span_extractor = SpanExtractor(tokenizer)

        self.index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def embed(self, sentence: str, target: str) -> np.ndarray:
        """Tạo contextual embedding cho target trong sentence."""
        normed_sentence=text_normalize(sentence)
        tokenized = self.tokenizer(normed_sentence, return_tensors="pt").to(self.device)
        span_idx = self.span_extractor.get_span_indices(sentence, target)
        span = torch.tensor(span_idx).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vec = self.model(tokenized, span)
        return vec.detach().cpu().numpy()

    def retrieve(self, sentence: str, target: str, top_k: int = 10, threshold: float = 0.0) -> list[dict]:
        vec = self.embed(sentence, target)
        distances, indices = self.index.search(vec, k=top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1.0 / (1.0 + float(dist))
            if similarity >= threshold:
                entry = dict(self.metadata[idx])
                entry["similarity"] = round(similarity, 4)
                results.append(entry)

        return results




class ViSLPipeline:
    """
    Pipeline:
      step 1 → 2: Normalize
      step 3:     Text-to-Gloss
      step 4:     Word Segmentation
      step 5-6:   Embedding + Vector DB Retrieval
    """

    def __init__(
        self,
        poses_path,
        embedding_model=None,
        embedding_tokenizer=None,
        faiss_index_path: str = None,
        metadata_path: str = None
    ):
        self.segmenter = WordSegmenter()
        self.poses_path = poses_path
        self.retriever = None
        if all([embedding_model, embedding_tokenizer, faiss_index_path, metadata_path]):
            self.retriever = EmbeddingRetriever(
                embedding_model=embedding_model,
                tokenizer=embedding_tokenizer,
                faiss_index_path=faiss_index_path,
                metadata_path=metadata_path,
            )

    def step2_normalize(self, text: str) -> str:
        result = normalize_text(text)
        return result

    def step3_gloss(self, text: str) -> dict | None:
        result = text_to_gloss(text)
        return result

    def step4_segment(self, gloss: dict) -> list[str]:
        flat_text = " ".join(gloss_to_token_list(gloss))
        tokens = self.segmenter.segment(flat_text)
        return tokens

    def step5_6_retrieve(
        self, original_sentence: str, tokens: list[str], top_k: int = 5, threshold: float = 0.0
    ) -> dict[str, list[dict]]:
        if self.retriever is None:
            print("[step 5-6] EmbeddingRetriever has not initialized yet.")
            return {}

        results = {}
        for token in tokens:
            try:
                hits = self.retriever.retrieve(original_sentence, token, top_k=top_k, threshold=threshold)
                results[token] = hits
            except ValueError as e:
                print(f"[step 5-6 - Retrieve] token error '{token}': {e}")
                results[token] = []
        return results

    def step_7_skeleton_generation_and_pose_smoothing(self,retrievals):
        best_retrieval = {
            k: v[0] for k, v in retrievals.items() if len(v) > 0
        }
        print(best_retrieval)

        pose_paths = [f"{self.poses_path}{v['Path']}.pose" for v in best_retrieval.values()]
        print(pose_paths)
        concat_pose = gloss_to_pose(pose_paths)
        return concat_pose

    def run(self, input_text: str, output_path, top_k: int = 5) -> dict:

        normalized   = self.step2_normalize(input_text)
        gloss        = self.step3_gloss(normalized)

        if gloss is None:
            print("[Pipeline] Cannot create gloss. stop pipeline.")
            return {}
        tokens       = self.step4_segment(gloss)
        concat_tokens =" ".join(tokens)

        retrievals   = self.step5_6_retrieve(normalized, tokens, top_k=top_k)

        concatenated_pose = self.step_7_skeleton_generation_and_pose_smoothing(retrievals)

        from pose_format.pose_visualizer import PoseVisualizer
        v = PoseVisualizer(concatenated_pose)
        v.save_video(output_path, v.draw())

        return {
            "input":       input_text,
            "normalized":  normalized,
            "gloss":       gloss,
            "tokens":      tokens,
            "retrievals":  retrievals,
        }

