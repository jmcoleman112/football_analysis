# python
from tracking.abstract_tracker import AbstractTracker

import cv2
import supervision as sv
from typing import List
from ultralytics.engine.results import Results
import numpy as np

class KeypointsTracker(AbstractTracker):
    """Detection and Tracking of football field keypoints"""

    def __init__(self, model_path: str, conf: float = 0.1, kp_conf: float = 0.7, infer_size: int = 680) -> None:
        """
        Initialize KeypointsTracker for tracking keypoints.

        Args:
            model_path (str): Model path.
            conf (float): Confidence threshold for field detection.
            kp_conf (float): Confidence threshold for keypoints.
            infer_size (int): Size used for model inference (square). Should match training size (e.g. 680).
        """
        super().__init__(model_path, conf)
        self.kp_conf = kp_conf
        self.tracks = []
        self.cur_frame = 0
        self.infer_size = infer_size  # inference resize size (e.g. 680)
        # default original size; will be updated from first input frame in detect()
        self.original_size = (1920, 1080)
        self.scale_x = float(self.original_size[0]) / float(self.infer_size)
        self.scale_y = float(self.original_size[1]) / float(self.infer_size)

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        """
        Perform keypoint detection on multiple frames.

        Args:
            frames (List[np.ndarray]): List of frames for detection.

        Returns:
            List[Results]: Detected keypoints for each frame
        """
        if not frames:
            return []

        # Update original size from the first frame (width, height)
        h, w = frames[0].shape[:2]
        self.original_size = (w, h)
        self.scale_x = float(self.original_size[0]) / float(self.infer_size)
        self.scale_y = float(self.original_size[1]) / float(self.infer_size)

        # Adjust contrast and resize to inference size
        contrast_adjusted_frames = [self._preprocess_frame(frame) for frame in frames]

        # Use YOLOv8's batch predict method
        detections = self.model.predict(contrast_adjusted_frames, conf=self.conf)
        return detections

    def track(self, detection: Results) -> dict:
        """
        Perform keypoint tracking based on detections.

        Args:
            detection (Results): Detected keypoints for a single frame.

        Returns:
            dict: Dictionary containing tracks of the frame.
        """
        detection = sv.KeyPoints.from_ultralytics(detection)

        if not detection:
            return {}

        xy = detection.xy[0]
        confidence = detection.confidence[0]

        # Use self.infer_size for bounds checking (inference image size)
        filtered_keypoints = {
            i: (coords[0] * self.scale_x, coords[1] * self.scale_y)
            for i, (coords, conf) in enumerate(zip(xy, confidence))
            if conf > self.kp_conf
            and 0 <= coords[0] <= self.infer_size
            and 0 <= coords[1] <= self.infer_size
        }

        self.tracks.append(detection)
        self.cur_frame += 1

        return filtered_keypoints

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame by adjusting contrast and resizing to infer_size x infer_size.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            np.ndarray: The resized frame with adjusted contrast.
        """
        frame = self._adjust_contrast(frame)
        resized_frame = cv2.resize(frame, (self.infer_size, self.infer_size))
        return resized_frame

    def _adjust_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Adjust the contrast of the frame using Histogram Equalization.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            np.ndarray: The frame with adjusted contrast.
        """
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            frame_equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            frame_equalized = cv2.equalizeHist(frame)

        return frame_equalized
