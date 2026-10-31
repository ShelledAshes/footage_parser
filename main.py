import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor




# returns video object and its fps
def open_video(path):
    path = str(path)
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) # retrieving its fps 

    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video FPS: {path}")

    return cap, float(fps)

# returning the persons and profile to be matched against
def load_models(weights_path):
    weights_path = Path(weights_path)

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"OSNet checkpoint not found: {weights_path}"
        )

    torch.set_num_threads(4)

    detector = YOLO(DETECTOR_MODEL)

    extractor = FeatureExtractor(
        model_name=REID_MODEL,
        model_path=str(weights_path),
        device=DEVICE,
        verbose=False,
    )

    return detector, extractor

def create_profile(args):
    output_dir = Path(args.profile)
    crops_dir = output_dir / "reference_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    detector, extractor = load_models(args.reid_weights)
    cap, fps = open_video(args.video)

    sample_every = max(1, round(fps * args.sample_seconds))
    frame_idx = 0
    selected_id = None
    first_selected_frame = None
    embeddings = []
    crop_index = 0

    print("Processing reference video on CPU.")
    print("A window will appear when tracked people are detected.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            current_frame = frame_idx
            frame_idx += 1

            if current_frame % args.frame_stride != 0:
                continue

            result, detections = get_detections(
                detector, frame, args.conf
            )

            # Select the target using a visible tracking ID.
            if selected_id is None and detections:
                annotated = result.plot()
                cv2.imshow("Select target person", annotated)
                cv2.waitKey(0)
                cv2.destroyWindow("Select target person")

                available_ids = sorted({
                    d["track_id"] for d in detections
                })

                print("Visible tracking IDs:", available_ids)

                while True:
                    try:
                        choice = int(input(
                            "Enter the target track ID shown in the video: "
                        ))

                        if choice in available_ids:
                            selected_id = choice
                            first_selected_frame = current_frame
                            break

                        print("Enter one of the displayed IDs.")
                    except ValueError:
                        print("Please enter a numeric track ID.")

            if selected_id is None:
                continue

            target = next(
                (
                    d for d in detections
                    if d["track_id"] == selected_id
                ),
                None,
            )

            if target is None:
                continue

            should_sample = (
                current_frame == first_selected_frame
                or current_frame % sample_every == 0
            )

            if not should_sample:
                continue

            crop = target["crop"]

            try:
                embedding = extract_embedding(extractor, crop)
            except Exception as exc:
                print(f"Skipping unusable crop: {exc}")
                continue

            crop_index += 1
            crop_path = crops_dir / f"frame_{crop_index:04d}.jpg"
            cv2.imwrite(str(crop_path), crop)

            embeddings.append(embedding.tolist())

            print(
                f"Saved reference sample {crop_index} "
                f"at {current_frame / fps:.2f}s"
            )

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not embeddings:
        raise RuntimeError(
            "No usable reference samples. Check the video, "
            "person visibility, detector, and selected track."
        )

    profile_embedding = average_embedding(embeddings)

    profile_data = {
        "profile_id": output_dir.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference_video": Path(args.video).name,
        "model": REID_MODEL,
        "embedding_method": "normalized_mean",
        "sample_count": len(embeddings),
        "embedding": profile_embedding.tolist(),
        "sample_embeddings": embeddings,
    }

    profile_file = output_dir / "profile.json"
    profile_file.write_text(
        json.dumps(profile_data, indent=2),
        encoding="utf-8",
    )

    print(f"\nProfile saved: {profile_file}")
    print(f"Reference samples: {len(embeddings)}")




if __name__ == "__main__":
    main()