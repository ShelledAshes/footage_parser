import os
# from unittest import result

import cv2
# import torch
# import numpy as np

from detector import Detector
# from extractor import Extractor
from vprofile import Profile


class VideoAnalyser:
    video_path = ''
    detector = None
    extractor = None
    frame = None
    conf = 0.35
    profile_map = {}
    last_sample_frame = {}
    sample_interval = 1.0  # seconds

    # test_detection_map = {1 : [], 16 : []}

    def __init__(self, video_path, detector_type, frame=None, conf=0.35):
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Video path {video_path} does not exist.")
        
        self.video_path = video_path
        self.frame = frame
        self.conf = conf

        self.detector = Detector(detector_type=detector_type)
        # self.detector.load_results(frame, conf=conf)

        # self.extractor = Extractor()

        self.analyse()

    def analyse(self):
        cap, fps = self.open_video(self.video_path)
        frame_idx = 0

        frame_stride = 1

        sample_interval_frames = max(
            1, int(fps * self.sample_interval)
        )

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                current_frame = frame_idx
                frame_idx += 1


                if current_frame % frame_stride != 0:
                    continue

                detections = self.detector.get_detections(frame)
                if len(detections) == 0:
                    continue

                # iteration through the detections per frame
                for detection in detections:

                    track_id = detection["track_id"] # 1, 3, 4, 8, 9, 14, 16
                    # test_ids = [1, 16]
                    # if track_id not in test_ids:
                    #     continue


                    if track_id not in self.profile_map:

                        self.profile_map[track_id] = Profile(
                            id=track_id
                        )

                        self.last_sample_frame[track_id] = -1

                    profile = self.profile_map[track_id]

                    last_frame = self.last_sample_frame[track_id]

                    should_sample = (
                        last_frame == -1
                        or current_frame - last_frame
                        >= sample_interval_frames
                    )

                    if not should_sample:
                        continue

                    crop = detection.get("crop")

                    if crop is None or crop.size == 0:
                        continue

                    # Store detection information.
                    detection_data = detection.copy()

                    detection_data["frame_number"] = current_frame
                    detection_data["timestamp"] = current_frame / fps

                    profile.add_detection_data(detection_data)

                    self.last_sample_frame[track_id] = current_frame

        finally:
            cap.release()
            # cv2.destroyAllWindows()
            pass

    def save_profile_screenshots(self, path=None):
        for track_id, profile in self.profile_map.items():

            profile.save_profile_data(path)
            profile.save_screenshots(path)

    def open_video(self, path):
        path = str(path)
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            cap.release()
            raise RuntimeError(f"Invalid video FPS: {path}")

        return cap, float(fps)


    