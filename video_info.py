
import os
import cv2

class VideoInfo:

    def __init__(self, path):

        self.valid = True
        if not path or not os.path.exists(path):
            self.valid = False
            raise ValueError(f"Video path {path} does not exist.")

        self.path = str(path)

        self.cap = cv2.VideoCapture(self.path)

        if not self.cap.isOpened():
            self.valid = False
            print(f"Cannot open video: {path}")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not self.fps or self.fps <= 0:
            self.valid = False
            self.cap.release()
            print(f"Invalid video FPS: {path}")
            return

        self.conf = 0.35
        self.frame_stride = 1
        self.sample_interval = 1.0
        self.sample_interval_frames = max(1, int(self.fps * self.sample_interval)) # sample_every
        self.max_gap_seconds = 2

        pass

    def get_sample_interval_frames(self):
        return self.sample_interval_frames

    def get_max_gap_seconds(self):
        return self.max_gap_seconds

    def get_frame_stride(self):
        return self.frame_stride

    def get_fps(self):
        return self.fps

    def get_cap(self):
        return self.cap

    def get_path(self):
        return self.path