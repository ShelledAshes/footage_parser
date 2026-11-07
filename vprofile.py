import os
from pathlib import Path
import json
import time
import uuid
import cv2
import torch
import numpy as np

from extractor import Extractor
import extractor
# from sample import extract_embedding

PROFILE_DIR = "profiles"

class Profile:

    # id = None

    # detections = []

    # created_at = None
    # reference_video_id = None
    # model_id = None
    # embedding_method_id = None
    # sample_count = 0
    # embedding = None
    # sample_embeddings = None

    def __init__(self, id=None):
        self.detections = []

        self.id = id if id else str(uuid.uuid4())
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S"),
        self.reference_video_id = None
        # self.model_id = None
        self.embedding_method = "normalized_mean"
        self.sample_count = 0
        self.embedding = None
        self.sample_embeddings = None

        self.extractor = Extractor()

        pass

    def add_detection_data(self, detection_data):
        if detection_data:
            # self.detections.append(detection_data.copy())
            self.detections.append(detection_data.copy())
            # print('profile detections:', self, len(self.detections))
            pass

    def load(self):
        for detection in self.detections:
            # process detection data to create embeddings or other profile data
            pass

    def save_profile_data(self, path):

        profile_data = self.get_profile_data()

        profile_dir = None
        if not path or not os.path.exists(path):
            print(f"Directory {path} does not exist.")
            profile_dir = Path(os.path.join(PROFILE_DIR, self.get_profile_dir_name()))
            profile_dir.mkdir(parents=True, exist_ok=True)
        else:
            profile_dir = Path(path) / self.get_profile_dir_name()
            profile_dir.mkdir(parents=True, exist_ok=True)

        profile_file = profile_dir / "profile.json"
        profile_file.write_text(
            json.dumps(profile_data, indent=2),
            encoding="utf-8",
        )

        pass

        
    def save_screenshots(self, path):
        crops_dir = None
        if not path or not os.path.exists(path):
            print(f"Directory {path} does not exist.")
            crops_dir = Path(os.path.join(PROFILE_DIR, self.get_profile_dir_name(), "reference_images"))
            crops_dir.mkdir(parents=True, exist_ok=True)
        else:
            crops_dir = Path(path) / self.get_profile_dir_name() / "reference_images"

        # crops_dir = crops_dir / "reference_images"

        crop_index = 0

        for detection in self.detections:
            crop = detection.get("crop", None)
            if crop is None or crop.size == 0:
                continue
            
            if crop is not None:
                crop_index += 1
                crop_path = crops_dir / f"frame_{crop_index:04d}.jpg"
                cv2.imwrite(str(crop_path), crop)

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
    
    def get_profile_data(self):

        embeddings = self.get_profile_embeddings()

        return {
            "profile_id": self.id,
            "created_at": self.created_at,
            "reference_video": self.reference_video_id, # video path
            "model": self.extractor.get_model_name() if self.extractor else None,
            "embedding_method": self.embedding_method,
            "sample_count": embeddings.__len__() if embeddings else 0,
            "embedding": self.average_embedding(embeddings).tolist() if embeddings else None,
            "sample_embeddings": embeddings,
        }

    # def write_to_dir(self, path):
    #     if not path or not os.path.exists(path):
    #         print(f"Directory {path} does not exist.")
    #         path = Path(os.path.join(PROFILE_DIR, self.id)) # might need to append profile.json to the path
    #         path.write_text(
    #                 json.dumps(self.get_data(), indent=2),
    #                 encoding="utf-8",
    #             )

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