import os
from pathlib import Path
import json
import time
import uuid
import cv2
import torch
import numpy as np

from extractor import Extractor
from video_info import VideoInfo

# import extractor
# from sample import extract_embedding

PROFILE_DIR = "profiles"
REF_IMG_DIR = "reference_images"
PROFILE_JSON = "profile.json"

class Profile:
    def __init__(self, id=None, video_info=None):
        self.detections = []

        self.id = id if id else str(uuid.uuid4())
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.vid_info = video_info
        
        self.reference_video_id = None
        # self.model_id = None
        self.embedding_method = "normalized_mean"
        self.sample_count = 0
        # self.embedding = None
        self.sample_embeddings = None # append embedding while appending detection data

        self.profile_save_path = None
        self.screenshot_paths = []

        # self.fps = fps
        
        # video clip props
        self.clips_track = {
                        "first_seen": 'timestamp', # change timestamp value to default
                        "last_seen": 'timestamp',
                        "last_frame": None,
                        "segments": [],
                        "embeddings": [],
                        "best_similarity": -1.0,
                        "best_frame": "",
                        "sample_count": 0,
                    }
        

        self.extractor = Extractor()

    def load_from_json(self, path):
        profile_file = Path.cwd() / "data" / path / PROFILE_JSON

        if not profile_file.is_file():
            raise FileNotFoundError(f"Profile file not found: {profile_file}")

        pdata = json.loads(profile_file.read_text(encoding="utf-8"))

        self.created_at = pdata.get("created_at", '')
        self.vid_info = VideoInfo(pdata.get("video_path", ''))
        self.extractor = Extractor()
        self.embedding_method = pdata.get('embedding_method', '')
        self.sample_count = pdata.get('sample_count', 0)
        self.clips_track = pdata.get('clips_track', dict())
        self.screenshot_paths = pdata.get('screenshots', list())

        self.sample_embeddings = pdata.get('sample_embeddings', list()) # not saving while first profile cretation

        pass

    def get_id(self):
        return self.id

    def get_profile_saved_path(self):
        return self.profile_save_path

    def get_ss_paths(self):
        return self.screenshot_paths

    def add_detection_data(self, det, embd_extract):
        if det:
            # self.detections.append(detection_data.copy())
            self.detections.append(det)

            if embd_extract:
                crop = det["crop"]

                try:
                    embedding = self.extract_embedding(self.extractor.get_model(), crop)
                except Exception as exc:
                    print(f"Skipping track {det['track_id']} crop: {exc}")
                    return

                self.clips_track["embeddings"].append(embedding)
                self.clips_track["sample_count"] += 1
            
            pass

    def load(self):
        for detection in self.detections:
            # process detection data to create embeddings or other profile data
            pass

    def save_profile_data(self, path):

        profile_dir = None
        if not path:
            print(f"Invalid directory path given.")
            profile_dir = Path(os.path.join(PROFILE_DIR, self.get_profile_dir_name()))
        elif not os.path.exists(path):
            print(f"Directory {path} does not exist.")
            profile_dir = Path(os.path.join(path, self.get_profile_dir_name()))
            profile_dir.mkdir(parents=True, exist_ok=True)
        else:
            profile_dir = Path(path) / self.get_profile_dir_name()
            profile_dir.mkdir(parents=True, exist_ok=True)

        profile_data = self.get_profile_data()

        profile_file = profile_dir / "profile.json"
        profile_file.write_text(
            json.dumps(profile_data, indent=2),
            encoding="utf-8",
        )

        self.profile_save_path = str(profile_file)

        pass

        
    def save_screenshots(self, path):
        
        crops_dir = None
        if not path:
            print(f"Invalid crop directory path given.")
            crops_dir = Path(os.path.join(PROFILE_DIR, self.get_profile_dir_name(), REF_IMG_DIR))
        elif not os.path.exists(path):
            print(f"Directory {path} does not exist.")
            crops_dir = Path(os.path.join(path, self.get_profile_dir_name(), REF_IMG_DIR))
            crops_dir.mkdir(parents=True, exist_ok=True)
        else:
            crops_dir = Path(path) / self.get_profile_dir_name() / REF_IMG_DIR
            crops_dir.mkdir(parents=True, exist_ok=True)

        crop_index = 0

        for detection in self.detections:
            crop = detection.get("crop", None)
            if crop is None or crop.size == 0:
                continue
            
            if crop is not None:
                crop_index += 1
                crop_path = crops_dir / f"frame_{crop_index:04d}.jpg"
                cv2.imwrite(str(crop_path), crop)

                self.screenshot_paths.append(str(crop_path))

        pass

    def get_profile_dir_name(self):
        return "profile_" + str(self.id)

    def get_profile_detection(self):
        return self.detections  

    def get_profile_embeddings(self):

        embeddings = []
        for detection in self.detections:
            crop = detection.get("crop", None)
            if crop is not None and crop.size > 0:
                # embedding = self.extractor.get_model().extract(crop)
                try:
                    embedding = self.extract_embedding(self.extractor.get_model(), crop)
                except Exception as exc:
                    print(f"Skipping unusable crop: {exc}")
                    continue

                embeddings.append(embedding.tolist())

        return embeddings

    def get_clips_track_json(self):
        data = self.clips_track

        embeddings = data.pop("embeddings")
        data["embeddings"] = []

        for embedding in embeddings:
            data["embeddings"].append(embedding.tolist())

        return data

    def get_profile_data(self):

        embeddings = self.get_profile_embeddings()

        return {
            "profile_id": self.id,
            "created_at": self.created_at,
            # "reference_video": self.reference_video_id, # video path
            "video_path": self.vid_info.get_path(),
            "model": self.extractor.get_model_name() if self.extractor else None,
            "embedding_method": self.embedding_method,
            "sample_count": embeddings.__len__() if embeddings else 0,
            "embedding": self.average_embedding(embeddings).tolist() if embeddings else None,
            "sample_embeddings": embeddings,
            # "clips_track": self.clips_track
            "clips_track": self.get_clips_track_json(), # check whether convertion to json is needed for dict
            "screenshots" : self.screenshot_paths
        }

    def extract_embedding(self, extractor, crop_bgr): # change into embedding class
        # Torchreid's image transform expects RGB image data.
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            features = extractor([crop_rgb])

        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()

        features = np.asarray(features, dtype=np.float32)

        return self.normalize(features[0])

    def normalize(self, vector):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)

        if norm < 1e-12:
            raise ValueError("Cannot normalize a zero-length embedding.")

        return vector / norm

    def average_embedding(self, embeddings):
        return self.normalize(np.mean(np.stack(embeddings), axis=0))

    def cosine_similarity(self, a, b):
        return float(np.dot(self.normalize(a), self.normalize(b)))


    def save_video_clips(self, path='', padding_seconds=2.0):

        video_path = self.vid_info.get_path()
        output_dir = None

        if path:
            output_dir = Path.cwd() / "data" / path
        else:
            output_dir = Path.cwd() / "data" / str(self.vid_info.id) / self.get_profile_dir_name()

        # output_dir = Path(output_dir) / "matching_clips"
        # output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # fps = self.fps
        fps = self.vid_info.get_fps()
        results = self.get_clip_results()
        
        """
        Save full-frame video clips for candidate matching tracks.
    
        Each clip includes configurable footage before and after
        the detected appearance interval.
        """
    
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

                
                # row["segments"] = [[0, 119]]
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

                        pass
    
        finally:
            cap.release()
    
        print(f"\nTotal video clips saved: {saved_count}")


    def get_clip_results(self):

        results = []

        threshold = 0.55

        embeddings = self.clips_track['embeddings']
        if embeddings.__len__() == 0:
            return results

        reference_embedding = self.normalize(
                                self.average_embedding(embeddings).tolist()
                            )
        
        # combined = self.average_embedding(item["embeddings"])
        combined = self.average_embedding(embeddings)
        aggregate_score = self.cosine_similarity(
            reference_embedding, combined
        )

        results.append({
                    "track_id": self.id,
        
                    "first_seen_sec": round(self.clips_track["first_seen"], 3),
                    "last_seen_sec": round(self.clips_track["last_seen"], 3),
        
                    "sample_count": self.clips_track["sample_count"],
        
                    "similarity": round(aggregate_score, 4),
        
                    "best_frame_similarity": round(
                        self.clips_track["best_similarity"], 4
                    ),
        
                    "candidate_match": (
                        aggregate_score >= threshold
                    ),
        
                    "best_frame": self.clips_track["best_frame"],
        
                    # Frame intervals for video clip extraction.
                    "segments": self.clips_track["segments"],
        
                    "clips": [],
                })

        results.sort(
                key=lambda row: row["similarity"],
                reverse=True,
            )

        return results

        