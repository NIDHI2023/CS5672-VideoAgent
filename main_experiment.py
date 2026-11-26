"""Run caption-based VideoAgent on an MSVD-QA subset with local Qwen.

Input annotations: JSON list of {id, video_id, question, answer}, or JSON
mapping question_id -> record. Captions: JSON mapping video_id -> caption list
(one caption per aligned sampled frame). CLIP retrieval is only called when
additional frames are requested; its features must be prepared separately.
"""
import json
import logging
import os
import threading
import re
import string
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_general import get_from_cache, save_to_cache

ANNOTATIONS_PATH = Path(os.getenv("MSVD_ANNOTATIONS", "msvdqa_subset.json"))
CAPTIONS_PATH = Path(os.getenv("MSVD_CAPTIONS", "msvdqa_captions.json"))
OUTPUT_PATH = Path(os.getenv("MSVD_OUTPUT", "msvdqa_predictions.json"))
MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "256"))
# The local GPU model is used sequentially, not from worker threads.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler("msvdqa_videoagent.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# These may be populated by configure_model() from a Colab notebook.
_tokenizer = None
_llm = None
_model_id = None
_generation_lock = threading.Lock()


def configure_model(model, tokenizer, model_id=MODEL):
    """Reuse an already loaded Hugging Face Qwen model and tokenizer."""
    global _llm, _tokenizer, _model_id
    _llm = model.eval()
    _tokenizer = tokenizer
    _model_id = str(model_id)


def load_model(model_id=MODEL):
    """Lazy-load once if the notebook hasn't supplied a model."""
    global _llm, _tokenizer, _model_id
    if _llm is not None:
        if _model_id != str(model_id):
            raise ValueError(
                f"Loaded model {_model_id!r}, requested {model_id!r}. "
                "Use configure_model() to switch models."
            )
        return
    logger.info("Loading Qwen model: %s", model_id)
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    _model_id = str(model_id)


def _extract_json(text):
    """Return a JSON object from a completion, allowing markdown fences."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Extract the first complete top-level object using JSONDecoder.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No valid JSON object in model response: {text[:400]!r}")


def get_llm_response(system_prompt, prompt, json_format=True, model=MODEL):
    """Generate locally with Qwen. Preserve existing string/JSON cache interface."""
    key = json.dumps(
        ["huggingface-qwen", model, system_prompt, prompt, json_format,
         MAX_NEW_TOKENS, "greedy"], ensure_ascii=False,
    )
    cached = get_from_cache(key)
    if cached is not None:
        return cached

    if _llm is None:
        load_model(model)
    elif _model_id != str(model):
        raise ValueError(f"Configured model {_model_id!r} != requested {model!r}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    last_error = None
    with _generation_lock:
        for attempt in range(3):
            try:
                # Retry with a stronger formatting reminder if the model
                # returns malformed JSON on the first attempt.
                candidate_messages = list(messages)
                if attempt:
                    candidate_messages[-1] = {
                        "role": "user",
                        "content": prompt +
                        "\nIMPORTANT: Output only one valid JSON object. "
                        "No markdown, commentary, or reasoning.",
                    }
                inputs = _tokenizer.apply_chat_template(
                    candidate_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(_llm.device)
                with torch.inference_mode():
                    generated = _llm.generate(
                        input_ids=inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=_tokenizer.eos_token_id,
                    )
                completion = _tokenizer.decode(
                    generated[0, inputs.shape[-1]:], skip_special_tokens=True
                ).strip()
                if not completion:
                    raise ValueError("Qwen returned an empty response")
                if json_format:
                    completion = json.dumps(_extract_json(completion), ensure_ascii=False)
                save_to_cache(key, completion)
                return completion
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("Qwen format error attempt %d/3: %s", attempt + 1, exc)
    raise RuntimeError("Qwen failed to return valid JSON after three attempts") from last_error


def parse_json(text):
    if isinstance(text, (dict, list)):
        return text
    return json.loads(text)


def parse_text_find_answer(text):
    obj = parse_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("final_answer"), str):
        raise ValueError(f"Invalid answer response: {text!r}")
    return obj["final_answer"].strip()


def parse_text_find_confidence(text):
    obj = parse_json(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid confidence response: {text!r}")
    confidence = int(obj["confidence"])
    if confidence not in (1, 2, 3):
        raise ValueError(f"Confidence outside 1..3: {confidence}")
    return confidence


def normalize_answer(answer):
    if answer is None:
        return ""
    value = str(answer).lower().strip()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def evaluate(ground_truth, prediction):
    return int(normalize_answer(ground_truth) == normalize_answer(prediction))


def read_caption(captions, sample_idx):
    """Return a dictionary using the original 1-based VideoAgent index convention."""
    return {f"frame {idx}": captions[idx - 1] for idx in sample_idx}


def answer_prompt(question, caption, num_frames, initial=False):
    sampling = "initially sampled" if initial else "available"
    return f"""You are answering an open-ended question about a video.
The video is represented by {num_frames} temporally ordered frame captions,
extracted at approximately 1 fps.
Here are the {sampling} frame descriptions (frame indices start at 1):
{json.dumps(caption, ensure_ascii=False, indent=2)}

Question: {question}

Use only the provided visual evidence. Return the shortest accurate word or
short phrase, without explanation. If evidence is insufficient, provide the
best evidence-supported prediction.
Return one JSON object of the form: {{"final_answer": "short answer"}}"""


def ask_gpt_caption(question, caption, num_frames):
    prompt = answer_prompt(question, caption, num_frames, initial=True)
    result = get_llm_response("Respond with valid JSON.", prompt)
    return prompt, result


def ask_gpt_caption_step(question, caption, num_frames):
    prompt = answer_prompt(question, caption, num_frames)
    result = get_llm_response("Respond with valid JSON.", prompt)
    return prompt, result


def generate_final_answer(question, caption, num_frames):
    return get_llm_response(
        "Respond with valid JSON.",
        answer_prompt(question, caption, num_frames),
    )


def self_eval(previous_prompt, answer):
    prompt = f"""Assess whether the frame captions provide sufficient evidence
for answering the video question. Do not use unseen video information.

Original prompt:
{previous_prompt}

Proposed answer:
{answer}

Confidence scale:
1: Important visual information is missing.
2: Partial evidence; the answer is still uncertain.
3: Available evidence strongly supports the answer.

Return exactly one JSON object: {{"confidence": 1}}
Replace 1 with the appropriate integer in 1, 2, or 3."""
    return get_llm_response("Respond with valid JSON.", prompt)


def generate_description_step(question, caption, num_frames, segment_des):
    prompt = f"""A video is represented by {num_frames} temporally ordered
frames sampled at approximately 1 fps. Only the following captions are known:
{json.dumps(caption, ensure_ascii=False, indent=2)}

Question: {question}

The current observations may be insufficient. Identify which unsampled
intervals could contain evidence needed to answer the question.
Candidate intervals, indexed by segment_id:
{json.dumps(segment_des, indent=2)}

For each useful interval, propose a concise hypothetical visual description
of the evidence to retrieve using text-image similarity. Do not claim the
hypothesized content has been observed. Use only the segment IDs above;
"duration" must exactly match the selected interval string.

Return JSON of the form:
{{"frame_descriptions": [
  {{"segment_id": "1", "duration": "1-4", "description": "a person holding an object"}}
]}}
The example fields are schematic, not necessarily valid segments here."""
    return get_llm_response("Respond with valid JSON.", prompt)


def get_retrieval_segments(sample_idx):
    """Only offer intervals containing at least one unseen 1-fps frame."""
    return {
        i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
        for i in range(len(sample_idx) - 1)
        if sample_idx[i + 1] - sample_idx[i] > 1
    }


def retrieve_more_frames(question, video_id, caps, sample_idx):
    segments = get_retrieval_segments(sample_idx)
    if not segments:
        return sample_idx
    proposed = parse_json(
        generate_description_step(question, read_caption(caps, sample_idx),
                                  len(caps), segments)
    )
    if not isinstance(proposed, dict) or not isinstance(proposed.get("frame_descriptions"), list):
        raise ValueError("Qwen did not return a frame_descriptions list")
    descriptions = [
        d for d in proposed["frame_descriptions"]
        if isinstance(d, dict)
        and str(d.get("segment_id", "")).isdigit()
        and int(d["segment_id"]) in segments
        and d.get("duration") == segments[int(d["segment_id"])]
        and isinstance(d.get("description"), str)
    ]
    if not descriptions:
        return sample_idx

    # This import is deferred so you can test the API/annotation path before
    # implementing the MSVD-specific CLIP retrieval and preparing embeddings.
    from utils_clip import frame_retrieval_seg_ego

    extra = frame_retrieval_seg_ego(descriptions, video_id, sample_idx)
    if extra is None:
        extra = []
    indices = [int(idx) for idx in extra]
    if any(idx < 1 or idx > len(caps) for idx in indices):
        raise ValueError(f"Retrieval returned out-of-range frame indices: {indices}")
    return sorted(set(sample_idx).union(indices))


def run_one_question(question_id, ann, caps):
    video_id = str(ann["video_id"])
    question = str(ann["question"])
    label = str(ann["answer"])
    if not isinstance(caps, list) or not caps:
        raise ValueError(f"No frame captions for video {video_id}")

    n = len(caps)
    sample_idx = sorted(set(np.linspace(1, n, num=min(5, n), dtype=int).tolist()))

    prompt, answer_text = ask_gpt_caption(question, read_caption(caps, sample_idx), n)
    prediction = parse_text_find_answer(answer_text)
    confidence = parse_text_find_confidence(self_eval(prompt, answer_text))
    rounds = 0

    # Original VideoAgent permits up to two rounds of additional retrieval.
    for _ in range(2):
        if confidence >= 3 or not get_retrieval_segments(sample_idx):
            break

        expanded = retrieve_more_frames(question, video_id, caps, sample_idx)
        if expanded == sample_idx:
            logger.info("No new frames for question %s", question_id)
            break
        sample_idx = expanded
        rounds += 1
        current_caps = read_caption(caps, sample_idx)

        if rounds == 2:
            answer_text = generate_final_answer(question, current_caps, n)
        else:
            prompt, answer_text = ask_gpt_caption_step(question, current_caps, n)
        prediction = parse_text_find_answer(answer_text)
        if rounds == 1:
            confidence = parse_text_find_confidence(self_eval(prompt, answer_text))

    return {
        "question_id": str(question_id),
        "video_id": video_id,
        "question": question,
        "answer": prediction,
        "label": label,
        "corr": evaluate(label, prediction),
        "count_frame": len(sample_idx),
        "sample_idx": sample_idx,
        "retrieval_rounds": rounds,
    }


def load_annotations(path):
    with open(path, encoding="utf-8") as f:
        annotations = json.load(f)
    if isinstance(annotations, list):
        records = [(str(record.get("id", record.get("question_id", idx))), record)
                   for idx, record in enumerate(annotations)]
    elif isinstance(annotations, dict):
        records = [(str(qid), record) for qid, record in annotations.items()]
    else:
        raise TypeError("Annotations must be a JSON list or dictionary")
    if len({qid for qid, _ in records}) != len(records):
        raise ValueError("Duplicate question IDs in annotations")
    for qid, ann in records:
        if not isinstance(ann, dict) or not all(
            field in ann for field in ("video_id", "question", "answer")
        ):
            raise ValueError(f"Annotation {qid} lacks video_id/question/answer")
    return records


def main():
    annotations = load_annotations(ANNOTATIONS_PATH)
    with open(CAPTIONS_PATH, encoding="utf-8") as f:
        captions = json.load(f)
    if not isinstance(captions, dict):
        raise TypeError("Caption file must map video IDs to caption lists")

    # Resume from existing results. Save after every question: Colab can disconnect.
    results = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            results = json.load(f)
        if not isinstance(results, dict):
            raise TypeError("Existing output must be a JSON dictionary")
        logger.info("Resuming with %d saved question results", len(results))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for qid, ann in annotations:
        if qid in results and "corr" in results[qid]:
            continue
        vid = str(ann["video_id"])
        try:
            result = run_one_question(qid, ann, captions[vid])
        except Exception as exc:
            logger.exception("Question %s (video %s) failed", qid, vid)
            result = {
                "question_id": qid, "video_id": vid, "question": ann["question"],
                "label": ann["answer"], "error": str(exc),
            }
        results[qid] = result
        tmp = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.replace(tmp, OUTPUT_PATH)

    selected = [results[qid] for qid, _ in annotations if qid in results]
    successful = [r for r in selected if "corr" in r]
    accuracy = sum(r["corr"] for r in successful) / len(successful) if successful else None
    print(f"Saved {len(selected)} questions to {OUTPUT_PATH}")
    print(f"Successful: {len(successful)}; errors: {len(selected) - len(successful)}")
    print(f"Exact-match accuracy on successful questions: {accuracy}")


if __name__ == "__main__":
    main()
