# python
from .abstract_annotator import AbstractAnnotator
from .abstract_video_processor import AbstractVideoProcessor
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper
from speed_estimation import SpeedEstimator
from .frame_number_annotator import FrameNumberAnnotator
from file_writing import TracksJsonWriter
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner
from ball_to_player_assignment import BallToPlayerAssigner
from utils import rgb_bgr_converter

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


class FootballVideoProcessor(AbstractAnnotator, AbstractVideoProcessor):
    """
    A video processor for football footage that tracks objects and keypoints,
    estimates speed, assigns the ball to player, calculates the ball possession
    and adds various annotations.

    New options:
        draw_heatmap (bool): enable overlaying a heatmap accumulated from tracks.
        heatmap_opacity (float): blending opacity for the heatmap overlay (0..1).
        heatmap_colormap (int): OpenCV colormap id used to colorize heatmap.
    """

    def __init__(self, obj_tracker: ObjectTracker, kp_tracker: KeypointsTracker,
                 club_assigner: ClubAssigner, ball_to_player_assigner: BallToPlayerAssigner,
                 top_down_keypoints: np.ndarray, field_img_path: str,
                 save_tracks_dir: Optional[str] = None, draw_frame_num: bool = True,
                 draw_heatmap: bool = False, heatmap_opacity: float = 0.6,
                 heatmap_colormap: int = cv2.COLORMAP_JET) -> None:

        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.club_assigner = club_assigner
        self.ball_to_player_assigner = ball_to_player_assigner
        self.projection_annotator = ProjectionAnnotator()
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints)
        self.draw_frame_num = draw_frame_num
        if self.draw_frame_num:
            self.frame_num_annotator = FrameNumberAnnotator()

        if save_tracks_dir:
            self.save_tracks_dir = save_tracks_dir
            self.writer = TracksJsonWriter(save_tracks_dir)

        field_image = cv2.imread(field_img_path)
        # Convert the field image to grayscale (black and white)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale back to 3 channels (since the main frame is 3-channel)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_GRAY2BGR)

        # Initialize the speed estimator with the field image's dimensions
        self.speed_estimator = SpeedEstimator(field_image.shape[1], field_image.shape[0])

        self.frame_num = 0

        self.field_image = field_image

        # Heatmap settings
        self.draw_heatmap = draw_heatmap
        self.heatmap_opacity = float(np.clip(heatmap_opacity, 0.0, 1.0))
        self.heatmap_colormap = heatmap_colormap
        self._heatmap_accum: Optional[np.ndarray] = None  # float32 accumulator

    def process(self, frames: List[np.ndarray], fps: float = 1e-6) -> List[np.ndarray]:
        self.cur_fps = max(fps, 1e-6)

        # Detect objects and keypoints in all frames
        batch_obj_detections = self.obj_tracker.detect(frames)
        batch_kp_detections = self.kp_tracker.detect(frames)

        processed_frames = []

        # Process each frame in the batch
        for idx, (frame, object_detection, kp_detection) in enumerate(zip(frames, batch_obj_detections, batch_kp_detections)):

            # Track detected objects and keypoints
            obj_tracks = self.obj_tracker.track(object_detection)
            kp_tracks = self.kp_tracker.track(kp_detection)

            # Assign clubs to players based on their tracked position
            obj_tracks = self.club_assigner.assign_clubs(frame, obj_tracks)

            all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

            # Map objects to a top-down view of the field
            all_tracks = self.obj_mapper.map(all_tracks)

            # Assign the ball to the closest player and calculate speed
            all_tracks['object'], _ = self.ball_to_player_assigner.assign(
                all_tracks['object'], self.frame_num,
                all_tracks['keypoints'].get(8, None),  # keypoint for player 1
                all_tracks['keypoints'].get(24, None)  # keypoint for player 2
            )

            # Estimate the speed of the tracked objects
            all_tracks['object'] = self.speed_estimator.calculate_speed(
                all_tracks['object'], self.frame_num, self.cur_fps
            )

            # Save tracking information if saving is enabled
            if self.save_tracks_dir:
                self._save_tracks(all_tracks)

            self.frame_num += 1

            # Annotate the current frame with the tracking information
            annotated_frame, projection_frame = self.annotate(frame, all_tracks)
            # Append a tuple (camera_frame, projection_frame) so callers can handle them separately
            processed_frames.append((annotated_frame, projection_frame))

        return processed_frames

    def process_frame(self, frame: np.ndarray, fps: float = 30.0) -> np.ndarray:
        self.cur_fps = max(fps, 1e-6)

        # Detect objects and keypoints in the single frame
        obj_detection = self.obj_tracker.detect([frame])[0]
        kp_detection = self.kp_tracker.detect([frame])[0]

        # Track detected objects and keypoints
        obj_tracks = self.obj_tracker.track(obj_detection)
        kp_tracks = self.kp_tracker.track(kp_detection)

        # Assign clubs to players based on their tracked position
        obj_tracks = self.club_assigner.assign_clubs(frame, obj_tracks)

        all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

        # Map objects to a top-down view of the field
        all_tracks = self.obj_mapper.map(all_tracks)

        # Assign the ball to the closest player and calculate speed
        all_tracks['object'], _ = self.ball_to_player_assigner.assign(
            all_tracks['object'], self.frame_num,
            all_tracks['keypoints'].get(8, None),
            all_tracks['keypoints'].get(24, None)
        )

        # Estimate the speed of the tracked objects
        all_tracks['object'] = self.speed_estimator.calculate_speed(
            all_tracks['object'], self.frame_num, self.cur_fps
        )

        # Save tracking information if saving is enabled
        if self.save_tracks_dir:
            self._save_tracks(all_tracks)

        self.frame_num += 1

        # Annotate the current frame with the tracking information
        annotated_frame, projection_frame = self.annotate(frame, all_tracks)

        # Return both frames so the caller (e.g. app) can display them separately
        return annotated_frame, projection_frame

    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        # Prepare annotated camera frame (no projection compositing)
        camera_frame = frame.copy()
        if self.draw_frame_num:
            camera_frame = self.frame_num_annotator.annotate(camera_frame, {'frame_num': self.frame_num})

        camera_frame = self.kp_annotator.annotate(camera_frame, tracks['keypoints'])
        camera_frame = self.obj_annotator.annotate(camera_frame, tracks['object'])

        # Generate projection according to current processor.projection_mode (defaults to 'heatmap')
        proj_mode = getattr(self, 'projection_mode', 'heatmap')
        projection_frame = self.projection_annotator.annotate(self.field_image, tracks['object'], mode=proj_mode)

        # If heatmap is enabled, keep heatmap on the camera frame (separate concern)
        if self.draw_heatmap:
            heat = self._generate_frame_heatmap(camera_frame, tracks)
            if heat is not None:
                camera_frame = self._overlay_heatmap(camera_frame, heat)

        # Return both images separately (camera frame, projection frame)
        return camera_frame, projection_frame


    def _combine_frame_projection(self, frame: np.ndarray, projection_frame: np.ndarray) -> np.ndarray:
        # Target canvas size
        canvas_width, canvas_height = 1920, 1080

        # Get dimensions of the original frame and projection frame
        h_frame, w_frame, _ = frame.shape
        h_proj, w_proj, _ = projection_frame.shape

        # Scale the projection to 70% of its original size
        scale_proj = 0.7
        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)
        projection_resized = cv2.resize(projection_frame, (new_w_proj, new_h_proj))

        # Create a blank canvas of 1920x1080
        combined_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Copy the main frame onto the canvas (top-left corner)
        combined_frame[:h_frame, :w_frame] = frame

        # Set the position for the projection frame at the bottom-middle
        x_offset = (canvas_width - new_w_proj) // 2
        y_offset = canvas_height - new_h_proj - 25  # 25px margin from bottom

        # Blend the projection with 75% visibility (alpha transparency)
        alpha = 0.75
        overlay = combined_frame[y_offset:y_offset + new_h_proj, x_offset:x_offset + new_w_proj]
        cv2.addWeighted(projection_resized, alpha, overlay, 1 - alpha, 0, overlay)

        return combined_frame


    def _annotate_possession(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()
        overlay = frame.copy()

        # Position and size for the possession overlay (top-left with 20px margin)
        overlay_width = 500
        overlay_height = 100
        gap_x = 20  # 20px from the left
        gap_y = 20  # 20px from the top

        # Draw background rectangle (black with transparency)
        cv2.rectangle(overlay, (gap_x, gap_y), (gap_x + overlay_width, gap_y + overlay_height), (0, 0, 0), -1)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Position for possession text
        text_x = gap_x + 15
        text_y = gap_y + 30

        # Display "Possession" above the progress bar
        cv2.putText(frame, 'Possession:', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)

        # Position and size for the possession bar (20px margin)
        bar_x = text_x
        bar_y = text_y + 25
        bar_width = overlay_width - bar_x
        bar_height = 15

        # Get possession data from the ball-to-player assigner
        possessions = self.ball_to_player_assigner.get_ball_possessions()
        if len(possessions) > 0:
            possession = possessions[-1]
            possession_club1 = possession[0]
            possession_club2 = possession[1]
        else:
            # Default to 50-50 possession if no data available yet
            possession_club1 = 0.5
            possession_club2 = 0.5

        # Calculate sizes for each possession segment in pixels
        club1_width = int(bar_width * possession_club1)
        club2_width = int(bar_width * possession_club2)
        neutral_width = bar_width - club1_width - club2_width

        club1_color = self.club_assigner.club1.player_jersey_color
        club2_color = self.club_assigner.club2.player_jersey_color
        neutral_color = (128, 128, 128)

        # Convert Club Colors from RGB to BGR
        club1_color = rgb_bgr_converter(club1_color)
        club2_color = rgb_bgr_converter(club2_color)

        # Draw club 1's possession (left)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + club1_width, bar_y + bar_height), club1_color, -1)

        # Draw neutral possession (middle)
        cv2.rectangle(frame, (bar_x + club1_width, bar_y), (bar_x + club1_width + neutral_width, bar_y + bar_height), neutral_color, -1)

        # Draw club 2's possession (right)
        cv2.rectangle(frame, (bar_x + club1_width + neutral_width, bar_y), (bar_x + bar_width, bar_y + bar_height), club2_color, -1)

        # Draw outline for the entire progress bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)

        # Calculate the position for the possession text under the bars
        possession_club1_text = f'{int(possession_club1 * 100)}%'
        possession_club2_text = f'{int(possession_club2 * 100)}%'

        # Display possession percentages for each club
        self._display_possession_text(frame, club1_width, club2_width, neutral_width, bar_x, bar_y, possession_club1_text, possession_club2_text, club1_color, club2_color)

        return frame


    def _display_possession_text(self, frame: np.ndarray, club1_width: int, club2_width: int,
                                  neutral_width: int, bar_x: int, bar_y: int,
                                 possession_club1_text: str, possession_club2_text: str,
                                 club1_color: Tuple[int, int, int], club2_color: Tuple[int, int, int]) -> None:
        # Text for club 1
        club1_text_x = bar_x + club1_width // 2 - 10  # Center of club 1's possession bar
        club1_text_y = bar_y + 35  # 20 pixels below the bar
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club1_color, 1)  # Club 1's color

        # Text for club 2
        club2_text_x = bar_x + club1_width + neutral_width + club2_width // 2 - 10  # Center of club 2's possession bar
        club2_text_y = bar_y + 35  # 20 pixels below the bar
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club2_color, 1)  # Club 2's color



    def _save_tracks(self, all_tracks: Dict[str, Dict[int, np.ndarray]]) -> None:
        self.writer.write(self.writer.get_object_tracks_path(), all_tracks['object'])
        self.writer.write(self.writer.get_keypoints_tracks_path(), all_tracks['keypoints'])


    # ---------------- Heatmap helpers ----------------

    def _generate_frame_heatmap(self, frame: np.ndarray, tracks: Dict) -> Optional[np.ndarray]:
        """
        Update and return the internal heatmap accumulator (single-channel float32)
        based on tracks. Attempts to use `top_down`, then `bbox`, then `center`
        keys in each object track entry. Returns accumulator or None on failure.
        """
        if not isinstance(frame, np.ndarray):
            return None

        h, w = frame.shape[:2]
        if self._heatmap_accum is None or self._heatmap_accum.shape != (h, w):
            self._heatmap_accum = np.zeros((h, w), dtype=np.float32)

        # create a temporary heatmap to add for this frame (float32)
        add_map = np.zeros((h, w), dtype=np.float32)

        object_tracks = tracks.get('object', {})
        for t in object_tracks.values():
            pos = None
            if isinstance(t, dict):
                td = t.get('top_down')
                if td is not None and len(td) >= 2:
                    # assume (x,y) coordinates
                    x, y = int(td[0]), int(td[1])
                    pos = (x, y)
                else:
                    bb = t.get('bbox')
                    if bb is not None and len(bb) >= 4:
                        # bbox format [x1,y1,x2,y2]
                        x = int((bb[0] + bb[2]) / 2)
                        y = int((bb[1] + bb[3]) / 2)
                        pos = (x, y)
                    else:
                        c = t.get('center')
                        if c is not None and len(c) >= 2:
                            x, y = int(c[0]), int(c[1])
                            pos = (x, y)

            if pos is not None and 0 <= pos[0] < w and 0 <= pos[1] < h:
                # draw a small gaussian spot on add_map
                rr = max(6, int(min(w, h) * 0.01))  # radius relative to frame size
                cv2.circle(add_map, (pos[0], pos[1]), rr, color=1.0, thickness=-1)

        # blur newly added spots for softer heat
        add_map = cv2.GaussianBlur(add_map, (0, 0), sigmaX=rr*1.5)

        # accumulate with a small weight and apply decay so older events fade
        self._heatmap_accum = cv2.add(self._heatmap_accum * 0.98, add_map, dtype=cv2.CV_32F)

        return self._heatmap_accum

    def _overlay_heatmap(self, frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """
        Convert single-channel heatmap to colored overlay and blend with frame.
        """
        if heatmap is None:
            return frame

        h, w = frame.shape[:2]
        if heatmap.shape != (h, w):
            heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap

        minv, maxv = float(heatmap_resized.min()), float(heatmap_resized.max())
        if maxv - minv < 1e-6:
            return frame

        norm = ((heatmap_resized - minv) / (maxv - minv) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(norm, self.heatmap_colormap)

        # Blend colored heatmap onto frame
        overlay = cv2.addWeighted(frame, 1.0 - self.heatmap_opacity, colored, self.heatmap_opacity, 0)
        return overlay
