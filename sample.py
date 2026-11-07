
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
# from torchreid import FeatureExtractor
from torchreid.reid.utils import FeatureExtractor


DETECTOR_MODEL = "yolo26n.pt"
REID_MODEL = "osnet_x0_5"
DEVICE = "cpu"


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero-length embedding.")

    return vector / norm


def cosine_similarity(a, b):
    return float(np.dot(normalize(a), normalize(b)))


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


def get_detections(detector, frame, conf=0.35):
    results = detector.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],          # COCO class 0 = person
        conf=conf,
        device=DEVICE,
        verbose=False,
    )

    result = results[0]
    detections = []

    if result.boxes is None or result.boxes.id is None:
        return result, detections

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().numpy()

    height, width = frame.shape[:2]

    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        # Reject tiny detections that are unlikely to yield
        # useful appearance features.
        if x2 - x1 < 32 or y2 - y1 < 64:
            continue

        detections.append({
            "track_id": int(track_id),
            "box": (x1, y1, x2, y2),
            "crop": frame[y1:y2, x1:x2].copy(),
        })

    return result, detections


def extract_embedding(extractor, crop_bgr):
    # Torchreid's image transform expects RGB image data.
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        features = extractor([crop_rgb])

    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()

    features = np.asarray(features, dtype=np.float32)
    return normalize(features[0])


def average_embedding(embeddings):
    return normalize(np.mean(np.stack(embeddings), axis=0))


def open_video(path):
    path = str(path)
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video FPS: {path}")

    return cap, float(fps)


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

def save_video_clips(
    video_path,
    output_dir,
    results,
    fps,
    padding_seconds=2.0,
):
    """
    Save full-frame video clips for candidate matching tracks.

    Each clip includes configurable footage before and after
    the detected appearance interval.
    """

    output_dir = Path(output_dir) / "matching_clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot reopen video: {video_path}"
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    padding_frames = round(padding_seconds * fps)

    saved_count = 0

    try:
        for row in results:

            # Save clips only for candidate matches.
            if not row["candidate_match"]:
                continue

            track_id = row["track_id"]
            row["clips"] = []

            for segment_index, segment in enumerate(
                row["segments"], start=1
            ):

                start_frame, end_frame = segment

                # Include footage before and after appearance.
                clip_start = max(
                    0, start_frame - padding_frames
                )

                clip_end = min(
                    total_frames - 1,
                    end_frame + padding_frames
                )

                filename = (
                    f"track_{track_id}_"
                    f"segment_{segment_index:03d}.mp4"
                )

                output_path = output_dir / filename

                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

                if not writer.isOpened():
                    print(
                        f"Could not create video: {output_path}"
                    )
                    writer.release()
                    continue

                try:
                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        clip_start
                    )

                    current_frame = clip_start

                    while current_frame <= clip_end:

                        ok, frame = cap.read()

                        if not ok:
                            break

                        writer.write(frame)
                        current_frame += 1

                finally:
                    writer.release()

                if output_path.is_file() and output_path.stat().st_size > 0:
                    saved_count += 1
                    row["clips"].append(str(output_path))

                    print(
                        f"Saved clip: {output_path} "
                        f"({clip_start / fps:.2f}s - "
                        f"{clip_end / fps:.2f}s)"
                    )

    finally:
        cap.release()

    print(f"\nTotal video clips saved: {saved_count}")

def search_video(args):
    profile_file = Path(args.profile) / "profile.json"

    if not profile_file.is_file():
        raise FileNotFoundError(
            f"Profile file not found: {profile_file}"
        )

    profile = json.loads(
        profile_file.read_text(encoding="utf-8")
    )
    reference_embedding = normalize(profile["embedding"])

    detector, extractor = load_models(args.reid_weights)
    cap, fps = open_video(args.video)

    output_dir = Path(args.output)
    crops_dir = output_dir / "candidates"
    crops_dir.mkdir(parents=True, exist_ok=True)

    sample_every = max(1, round(fps * args.sample_seconds))
    frame_idx = 0
    tracks = {}

    print("Searching video on CPU.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            current_frame = frame_idx
            frame_idx += 1

            if current_frame % args.frame_stride != 0:
                continue

            timestamp = current_frame / fps

            _, detections = get_detections(
                detector, frame, args.conf
            )

            # Record track visibility even on frames where
            # we do not extract an expensive Re-ID embedding.
            for det in detections:
                tid = det["track_id"]

                # if tid not in tracks:
                #     tracks[tid] = {
                #         "first_seen": timestamp,
                #         "last_seen": timestamp,
                #         "embeddings": [],
                #         "best_similarity": -1.0,
                #         "best_frame": "",
                #         "sample_count": 0,
                #     }

                # tracks[tid]["last_seen"] = timestamp

                
                if tid not in tracks:
                    tracks[tid] = {
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                        "last_frame": None,
                        "segments": [],
                        "embeddings": [],
                        "best_similarity": -1.0,
                        "best_frame": "",
                        "sample_count": 0,
                    }

                item = tracks[tid]

                item["last_seen"] = timestamp

                # Track separate appearance intervals.
                max_gap_frames = round(args.max_gap_seconds * fps)

                if (
                    not item["segments"]
                    or current_frame - item["last_frame"] > max_gap_frames
                ):
                    # Start a new appearance interval.
                    item["segments"].append([
                        current_frame,
                        current_frame
                    ])
                else:
                    # Extend the existing interval.
                    item["segments"][-1][1] = current_frame

                item["last_frame"] = current_frame


            if current_frame % sample_every != 0:
                continue

            for det in detections:
                tid = det["track_id"]
                crop = det["crop"]

                try:
                    embedding = extract_embedding(
                        extractor, crop
                    )
                except Exception as exc:
                    print(
                        f"Skipping track {tid} crop: {exc}"
                    )
                    continue

                item = tracks[tid]
                item["embeddings"].append(embedding)
                item["sample_count"] += 1

                # Save the best individual crop for review.
                score = cosine_similarity(
                    reference_embedding, embedding
                )

                if score > item["best_similarity"]:
                    item["best_similarity"] = score

                    image_path = (
                        crops_dir / f"track_{tid}_best.jpg"
                    )
                    cv2.imwrite(str(image_path), crop)
                    item["best_frame"] = str(image_path)

    finally:
        cap.release()

    results = []

    for tid, item in tracks.items():
        if not item["embeddings"]:
            continue

        combined = average_embedding(item["embeddings"])
        aggregate_score = cosine_similarity(
            reference_embedding, combined
        )

        # results.append({
        #     "track_id": tid,
        #     "first_seen_sec": round(item["first_seen"], 3),
        #     "last_seen_sec": round(item["last_seen"], 3),
        #     "sample_count": item["sample_count"],
        #     "similarity": round(aggregate_score, 4),
        #     "best_frame_similarity": round(
        #         item["best_similarity"], 4
        #     ),
        #     "candidate_match": (
        #         aggregate_score >= args.threshold
        #     ),
        #     "best_frame": item["best_frame"],
        # })

        
        results.append({
            "track_id": tid,

            "first_seen_sec": round(item["first_seen"], 3),
            "last_seen_sec": round(item["last_seen"], 3),

            "sample_count": item["sample_count"],

            "similarity": round(aggregate_score, 4),

            "best_frame_similarity": round(
                item["best_similarity"], 4
            ),

            "candidate_match": (
                aggregate_score >= args.threshold
            ),

            "best_frame": item["best_frame"],

            # Frame intervals for video clip extraction.
            "segments": item["segments"],

            "clips": [],
        })


    results.sort(
        key=lambda row: row["similarity"],
        reverse=True,
    )

    save_video_clips(
        video_path=args.video,
        output_dir=output_dir,
        results=results,
        fps=fps,
        padding_seconds=args.clip_padding,
    )

    csv_path = output_dir / "matches.csv"
    # fieldnames = [
    #     "track_id",
    #     "first_seen_sec",
    #     "last_seen_sec",
    #     "sample_count",
    #     "similarity",
    #     "best_frame_similarity",
    #     "candidate_match",
    #     "best_frame",
    # ]

    fieldnames = [
        "track_id",
        "first_seen_sec",
        "last_seen_sec",
        "sample_count",
        "similarity",
        "best_frame_similarity",
        "candidate_match",
        "best_frame",
        "clips",
    ]

    for row in results:
        row["clips"] = ";".join(row["clips"])
        row.pop("segments", None)

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved: {csv_path}")
    print(f"Candidate tracks: {len(results)}")

    for row in results[:10]:
        status = (
            "CANDIDATE"
            if row["candidate_match"]
            else "below threshold"
        )

        print(
            f"Track {row['track_id']:>3} | "
            f"score={row['similarity']:.3f} | "
            f"{row['first_seen_sec']:.2f}s-"
            f"{row['last_seen_sec']:.2f}s | "
            f"{status}"
        )



def main():
    parser = argparse.ArgumentParser(
        description="CPU-based appearance person Re-ID"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True
    )

    def add_common_options(subparser):
        subparser.add_argument(
            "--reid-weights",
            # default="models/osnet_x0_5_market1501.pth.tar",
            default=(
                        "models/"
                        "osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
                    )
        )
        subparser.add_argument(
            "--sample-seconds",
            type=float,
            default=1.0,
            help="Minimum interval between embedding samples",
        )
        subparser.add_argument(
            "--frame-stride",
            type=int,
            default=1,
            help="Run detection every Nth decoded frame",
        )
        subparser.add_argument(
            "--conf",
            type=float,
            default=0.35,
            help="YOLO person detection confidence",
        )

    create_parser = subparsers.add_parser("create-profile")
    create_parser.add_argument("--video", required=True)
    create_parser.add_argument("--profile", required=True)
    add_common_options(create_parser)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--video", required=True)
    search_parser.add_argument("--profile", required=True)
    search_parser.add_argument("--output", default="output")
    search_parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Candidate ranking threshold; calibrate for your cameras",
    )
    search_parser.add_argument(
        "--clip-padding",
        type=float,
        default=2.0,
        help="Seconds of footage before and after each appearance",
    )

    search_parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=2.0,
        help="Maximum detection gap before starting a new clip",
    )

    add_common_options(search_parser)

    args = parser.parse_args()

    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be positive")

    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")

    # if args.clip_padding < 0:
    #     parser.error("--clip-padding cannot be negative")

    # if args.max_gap_seconds <= 0:
    #     parser.error("--max-gap-seconds must be positive")
        
    if args.command == "create-profile":
        create_profile(args)
    else:
        if not -1.0 <= args.threshold <= 1.0:
            parser.error("--threshold must be between -1 and 1")
        search_video(args)


if __name__ == "__main__":
    main()
