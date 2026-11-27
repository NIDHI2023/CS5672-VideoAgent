# reusable video download and frame extraction
import cv2
import numpy as np
from PIL import Image


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