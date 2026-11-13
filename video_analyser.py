import os
from pathlib import Path

import cv2
import base64
# import torch
# import numpy as np

from .detector import Detector
# from extractor import Extractor
from .vprofile import Profile
from .video_info import VideoInfo


class VideoAnalyser:

    # def __init__(self, video_path, detector_type, frame=None, conf=0.35):
    def __init__(self, id, video_path, detector_type):
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Video path {video_path} does not exist.")
        
        self.video_info = VideoInfo(video_path)
        self.data_path = Path.cwd() / "data" / str(id) # change to global saving directory
        # self.frame = frame
        # self.conf = conf
        # self.sample_interval = 1.0  # seconds
        # self.max_gap_seconds = 2

        self.last_sample_frame = {}
        self.profile_map = {}
        self.tracks = {}

        self.detector = Detector(detector_type=detector_type)

        self.analyse()

    def analyse(self):
        # cap, fps = self.open_video(self.video_path)
        cap = self.video_info.get_cap()
        fps = self.video_info.get_fps()

        frame_idx = 0
        # frame_stride = 1

        # sample_interval_frames = max( # sample_every
        #     1, int(fps * self.sample_interval)
        # )

        sample_interval_frames = self.video_info.get_sample_interval_frames()

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                current_frame = frame_idx
                frame_idx += 1

                # if current_frame % frame_stride != 0:
                if current_frame % self.video_info.get_frame_stride() != 0:
                    continue

                detections = self.detector.get_detections(frame)
                if len(detections) == 0:
                    continue

                extract_embeddings = (
                    current_frame % sample_interval_frames == 0
                )

                # iteration through the detections per frame
                for detection in detections:

                    track_id = detection["track_id"] # 1, 3, 4, 8, 9, 14, 16
                    # test_ids = [1]
                    # if track_id not in test_ids:
                    #     continue

                    timestamp = current_frame / fps # Video_clipping

                    if track_id not in self.profile_map:

                        created_profile = Profile(
                            id=track_id,
                            video_info=self.video_info
                        )

                        created_profile.clips_track["first_seen"] = timestamp # Video_clipping
                        created_profile.clips_track["last_seen"] = timestamp # Video_clipping

                        self.profile_map[track_id] = created_profile

                        self.last_sample_frame[track_id] = -1


                    profile = self.profile_map[track_id]

                    last_frame = self.last_sample_frame[track_id]

                    should_sample = ( # only for taking screenshots
                        last_frame == -1
                        or current_frame - last_frame
                        >= sample_interval_frames
                    )

                    # if not should_sample:
                    #     continue

                    crop = detection.get("crop")

                    if crop is None or crop.size == 0:
                        continue

                    # Store detection information.
                    detection_data = detection.copy()

                    detection_data["frame_number"] = current_frame
                    detection_data["timestamp"] = current_frame / fps

                    # profile.add_detection_data(detection_data)

                    self.last_sample_frame[track_id] = current_frame


                    '''''''''''/***********************/'''''''''''

                    profile.clips_track["last_seen"] = timestamp

                    # max_gap_frames = round(self.max_gap_seconds * fps)
                    max_gap_frames = round(self.video_info.get_max_gap_seconds() * fps)

                    if (
                        not profile.clips_track["segments"]
                        or current_frame - profile.clips_track["last_frame"] > max_gap_frames
                    ):
                        # Start a new appearance interval.
                        profile.clips_track["segments"].append([
                            current_frame,
                            current_frame
                        ])
                    else:
                        # Extend the existing interval.
                        profile.clips_track["segments"][-1][1] = current_frame

                    profile.clips_track["last_frame"] = current_frame

                    # ----------------------------------
                    # 2. Extract embedding only when needed
                    # ----------------------------------

                    # if not extract_embeddings:
                    #     continue

                    profile.add_detection_data(detection_data, extract_embeddings)

        finally:
            cap.release()
            # cv2.destroyAllWindows()
            pass

    def save_profile_screenshots(self, path=None):

        for track_id, profile in self.profile_map.items():
            profile.save_screenshots(self.data_path)
            profile.save_profile_data(self.data_path)

    def get_profiles_data(self):
        profiles_data = []

        for track_id, profile in self.profile_map.items():
            profiles_data.append({
                    "id": profile.get_id(),
                    "path" : profile.get_profile_saved_path()
                 })

        return profiles_data

    def get_profile_data_for_recognition(self):
        data = []

        for id, profile in self.profile_map.items():
            pdata = {
                "id" : profile.get_id(),
                "img_path": profile.get_ss_paths()[0]
            }

            # img_path = profile.get_ss_paths()[0]
            # with open(img_path, "rb") as image_file:    # instead of appending just image data, append filetype and filename, just in case
            #     image_data = base64.b64encode(image_file.read()).decode("utf-8")
            #     pdata["image_data"] = image_data

            data.append(pdata)

        return data


    def save_profile_clips(self, profile_id, fps=30):
        if profile_id > 0:
            profile = self.profile_map.get(profile_id, None)
            if profile:
                profile.save_video_clips(3)

    # def open_video(self, path):
    #     path = str(path)
    #     cap = cv2.VideoCapture(path)

    #     if not cap.isOpened():
    #         raise RuntimeError(f"Cannot open video: {path}")

    #     fps = cap.get(cv2.CAP_PROP_FPS)

    #     if not fps or fps <= 0:
    #         cap.release()
    #         raise RuntimeError(f"Invalid video FPS: {path}")

    #     return cap, float(fps)


    