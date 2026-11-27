import pickle
from typing import List

import numpy as np

import os
import pickle
from typing import List

import numpy as np

from clip_text_encoder import CLIPTextEncoder


ORIGINAL_CACHE_PATH = "cache_clip.pkl"
QWEN_CACHE_PATH = "cache_clip_qwen.pkl"

with open(ORIGINAL_CACHE_PATH, "rb") as f:
    cache_clip = pickle.load(f)

if os.path.exists(QWEN_CACHE_PATH):
    with open(QWEN_CACHE_PATH, "rb") as f:
        qwen_cache = pickle.load(f)
else:
    qwen_cache = {}

clip_encoder = None


def get_clip_encoder():
    global clip_encoder

    if clip_encoder is None:
        clip_encoder = CLIPTextEncoder()

    return clip_encoder


def save_qwen_cache():
    temp_path = QWEN_CACHE_PATH + ".tmp"

    with open(temp_path, "wb") as f:
        pickle.dump(qwen_cache, f)

    os.replace(temp_path, QWEN_CACHE_PATH)


def get_embeddings(inputs: List[str]) -> np.ndarray:

    missing = list(dict.fromkeys(
        text for text in inputs
        if text not in cache_clip
        and text not in qwen_cache
    ))

    print(
        f"[CLIP] Queries: {len(inputs)} | "
        f"New descriptions: {len(missing)}",
        flush=True
    )

    if missing:
        encoder = get_clip_encoder()
        embeddings = encoder.encode(missing)

        assert embeddings.shape == (
            len(missing), 1280
        )

        for text, embedding in zip(missing, embeddings):
            qwen_cache[text] = embedding

        save_qwen_cache()

    results = []

    for text in inputs:
        if text in cache_clip:
            embedding = cache_clip[text]
        else:
            embedding = qwen_cache[text]

        results.append(
            np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)
        )

    return np.stack(results)

def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
    frame_embeddings = np.load(f"ego_features_448/{video_id}.npy")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions]
    )
    frame_idx = []
    for idx, description in enumerate(descriptions):
        seg = int(description["segment_id"]) - 1
        seg_frame_embeddings = frame_embeddings[sample_idx[seg] : sample_idx[seg + 1]]
        if seg_frame_embeddings.shape[0] < 2:
            frame_idx.append(sample_idx[seg] + 1)
            continue
        seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)

    return frame_idx


if __name__ == "__main__":
    pass
