import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_clip import frame_retrieval_seg_ego
from utils_general import save_to_cache
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("egoschema_subset.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# initalize Qwen instead of OpenAI
MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/content/models/Qwen2.5-1.5B-Instruct"
)

FORCE_RETRIEVAL = int(os.environ.get(
    "FORCE_RETRIEVAL",
    "0"
))
if FORCE_RETRIEVAL:
  print(f"Using FORCE_RETRIEVAL: {FORCE_RETRIEVAL}")

OUTPUT_FILE = os.environ.get(
    "OUTPUT_FILE",
    "egoschema.json"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading Qwen from {MODEL_PATH}")
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

qwen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    local_files_only=True
).to(DEVICE)

qwen_model.eval()

print("Qwen model loaded successfully")

# improve parse_json since qwen could be inconsistent
import ast

def parse_json(text):
    if not isinstance(text, str):
        return None

    text = text.strip()

    # Parse complete JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Parse complete Python dictionary.
    try:
        result = ast.literal_eval(text)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass

    # Extract nested JSON from surrounding text.
    decoder = json.JSONDecoder()

    for match in re.finditer(r"[\{\[]", text):
        try:
            result, _ = decoder.raw_decode(
                text[match.start():]
            )

            if isinstance(result, (dict, list)):
                return result

        except json.JSONDecodeError:
            continue

    # Extract Python-style nested dictionaries.
    for start in re.finditer(r"[\{\[]", text):
        for end in reversed(
            list(re.finditer(r"[\}\]]", text))
        ):
            if end.start() <= start.start():
                continue

            candidate = text[start.start():end.end()]

            try:
                result = ast.literal_eval(candidate)

                if isinstance(result, (dict, list)):
                    return result

            except (ValueError, SyntaxError):
                continue

    logger.warning("No valid JSON found: %s", text)
    return None


def parse_text_find_number(text):
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        return -1


def parse_text_find_confidence(text):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            return random.randint(1, 3)
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1


def get_llm_response(
    system_prompt, prompt, json_format=True, model=MODEL_PATH
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]

    logger.info(messages)
    # Replace get from cache

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False
    ).to(DEVICE)

    input_length = inputs["input_ids"].shape[1]

    generated_ids = qwen_model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # Remove the input prompt from the generated sequence.
    response = tokenizer.decode(
        generated_ids[0, input_length:],
        skip_special_tokens=True
    ).strip()

    logger.info(response)

    if json_format and parse_json(response) is None:
        logger.warning("Qwen returned invalid JSON")

    return response


def generate_final_answer(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think carefully and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates.
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def generate_description_step(question, caption, num_frames, segment_des):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. 
    Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def self_eval(previous_prompt, answer):
    confidence_format = {"confidence": "xxx"}
    prompt = f"""Please assess the confidence level in the decision-making process.
    The provided information is as as follows,
    {previous_prompt}
    The decision making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def ask_gpt_caption(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of five uniformly sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def ask_gpt_caption_step(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def read_caption(captions, sample_idx):
    video_caption = {}
    for idx in sample_idx:
        video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption


def run_one_question(video_id, ann, caps, logs):
    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5)]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    num_frames = len(caps)

    ### Step 1 ###
    sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()
    sampled_caps = read_caption(caps, sample_idx)
    previous_prompt, answer_str = ask_gpt_caption(
        formatted_question, sampled_caps, num_frames
    )
    answer = parse_text_find_number(answer_str)
    confidence_str = self_eval(previous_prompt, answer_str)
    confidence = parse_text_find_confidence(confidence_str)

    ### Step 2 ###
    if confidence < 3 or FORCE_RETRIEVAL:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 2: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))

            sampled_caps = read_caption(caps, sample_idx)
            previous_prompt, answer_str = ask_gpt_caption_step(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)
            confidence_str = self_eval(previous_prompt, answer_str)
            confidence = parse_text_find_confidence(confidence_str)
        except Exception as e:
            logger.error(f"Step 2 Error: {e}")
            print(f"[Step 2] Retrieval failed: {e}", flush=True)
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)

    ### Step 3 ###
    if confidence < 3 or FORCE_RETRIEVAL:
        print("step3")
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 3: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))
            sampled_caps = read_caption(caps, sample_idx)
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)
        except Exception as e:
            logger.error(f"Step 3 Error: {e}")
            print(f"[Step 3] Retrieval failed: {e}", flush=True)
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)
    if answer == -1:
        logger.info("Answer Index Not Found!")
        answer = random.randint(0, 4)

    logger.info(f"Finished video: {video_id}/{answer}/{ann['truth']}")

    label = int(ann["truth"])
    corr = int(label == answer)
    count_frame = len(sample_idx)

    logs[video_id] = {
        "answer": answer,
        "label": label,
        "corr": corr,
        "count_frame": count_frame,
    }


def main():
    # if running full set, change subset to fullset
    data_path = Path("data") / "EgoSchema"
    input_ann_file = data_path /"subset_anno.json"
    all_cap_file = data_path / "lavila_subset.json"
    json_file_name = data_path / OUTPUT_FILE

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    max_questions = int(
        os.environ.get("MAX_QUESTIONS", "5")
    )
    print(f"Evaluating {max_questions if max_questions > 0 else "all"} questions...")

    anns = json.load(open(input_ann_file, "r"))
    all_caps = json.load(open(all_cap_file, "r"))
    logs = {}

    video_ids = list(anns.keys())

    if max_questions > 0:
        video_ids = video_ids[:max_questions]

    for i, video_id in enumerate(video_ids):

        print(
            f"\n[{i + 1}/{len(video_ids)}] "
            f"Processing {video_id}",
            flush=True
        )

        run_one_question(
            video_id,
            anns[video_id],
            all_caps[video_id],
            logs
        )

        result = logs[video_id]

        print(
            f"Prediction: {result['answer']} | "
            f"Ground truth: {result['label']} | "
            f"Frames used: {result['count_frame']}",
            flush=True
        )

        # Save after every video.
        with open(json_file_name, "w") as f:
            json.dump(logs, f, indent=2)

    accuracy = np.mean([
        result["corr"] for result in logs.values()
    ])

    mean_frames = np.mean([
        result["count_frame"] for result in logs.values()
    ])

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Mean frames: {mean_frames:.2f}")
    print(f"Results saved to {json_file_name}")


if __name__ == "__main__":
    main()