"""Run caption-based VideoAgent on an MSVD-QA subset with Gemini.

Input annotations: JSON list of {id, video_id, question, answer}, or JSON
mapping question_id -> record. Captions: JSON mapping video_id -> caption list
(one caption per aligned sampled frame). CLIP retrieval is only called when
additional frames are requested; its features must be prepared separately.
"""
import json
import logging
import os
import re
import string
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from utils_general import get_from_cache, save_to_cache

ANNOTATIONS_PATH = Path(os.getenv("MSVD_ANNOTATIONS", "msvdqa_subset.json"))
CAPTIONS_PATH = Path(os.getenv("MSVD_CAPTIONS", "msvdqa_captions.json"))
OUTPUT_PATH = Path(os.getenv("MSVD_OUTPUT", "msvdqa_predictions.json"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler("msvdqa_videoagent.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def make_client():
    """Use Vertex AI if configured, otherwise use a Gemini API key."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project:
        return genai.Client(
            vertexai=True,
            project=project,
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        )
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    raise RuntimeError(
        "Set GOOGLE_CLOUD_PROJECT (Vertex AI with ADC) or GEMINI_API_KEY."
    )


client = None


def get_llm_response(system_prompt, prompt, json_format=True, model=MODEL):
    """Return text, using the existing cache. Fail visibly if Gemini fails."""
    global client
    key = json.dumps(
        ["gemini", model, system_prompt, prompt, json_format],
        ensure_ascii=False,
    )
    cached = get_from_cache(key)
    if cached is not None:
        return cached

    if client is None:
        client = make_client()

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0,
        response_mime_type="application/json" if json_format else "text/plain",
    )
    last_error = None
    for attempt in range(3):
        try:
            completion = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            result = completion.text
            if not result:
                raise ValueError("Gemini returned an empty response")
            if json_format:
                json.loads(result)  # Reject invalid JSON instead of caching it.
            save_to_cache(key, result)
            return result
        except Exception as exc:
            last_error = exc
            logger.warning("Gemini request failed, attempt %d/3: %s", attempt + 1, exc)
    raise RuntimeError("Gemini failed after three attempts") from last_error


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
        raise ValueError("Gemini did not return a frame_descriptions list")
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

    def process(item):
        qid, ann = item
        vid = str(ann["video_id"])
        try:
            return qid, run_one_question(qid, ann, captions[vid])
        except Exception as exc:
            logger.exception("Question %s (video %s) failed", qid, vid)
            return qid, {
                "question_id": qid, "video_id": vid, "question": ann["question"],
                "label": ann["answer"], "error": str(exc),
            }

    # Executor.map propagates results in the input order even with parallel workers.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = dict(executor.map(process, annotations))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    successful = [r for r in results.values() if "corr" in r]
    accuracy = sum(r["corr"] for r in successful) / len(successful) if successful else None
    print(f"Saved {len(results)} questions to {OUTPUT_PATH}")
    print(f"Successful: {len(successful)}; errors: {len(results) - len(successful)}")
    print(f"Exact-match accuracy on successful questions: {accuracy}")


if __name__ == "__main__":
    main()
