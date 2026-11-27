# reusable video download and frame extraction
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info

CAPTION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

caption_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CAPTION_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

caption_processor = AutoProcessor.from_pretrained(
    CAPTION_MODEL_ID,
    min_pixels=128 * 28 * 28,
    max_pixels=256 * 28 * 28,
)

print("GPU:", torch.cuda.get_device_name(0))

CAPTION_PROMPT = (
    "Describe this video frame in one concise factual sentence. "
    "Include the visible people, objects, actions, and important "
    "spatial relationships. Do not speculate about events outside "
    "the frame. Do not include introductory text."
)

@torch.inference_mode()
def caption_frame(frame_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Path(frame_path).resolve().as_uri(),
                },
                {
                    "type": "text",
                    "text": CAPTION_PROMPT,
                },
            ],
        }
    ]

    prompt = caption_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = caption_processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated = caption_model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

    generated = generated[:, inputs.input_ids.shape[1]:]

    return caption_processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def extract_frames(video_path, output_dir, sample_fps=1):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video: {video_path}")

    last_timestamp = (total_frames - 1) / fps

    timestamps = np.arange(
        0,
        last_timestamp + 1e-6,
        1 / sample_fps
    )

    records = []

    try:
        for timestamp in timestamps:
            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                float(timestamp * 1000)
            )

            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Could not decode {video_path.name} "
                    f"at {timestamp:.2f}s"
                )

            frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            path = output_dir / f"{len(records):04d}.jpg"
            Image.fromarray(frame).save(path, quality=90)

            records.append({
                "frame_index": len(records) + 1,
                "timestamp_seconds": float(timestamp),
                "path": str(path),
            })
    finally:
        cap.release()

    return records