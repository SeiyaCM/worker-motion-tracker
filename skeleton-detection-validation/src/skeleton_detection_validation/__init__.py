"""OpenCV and MediaPipe skeleton detection validation package."""

from .base_detector import BaseSkeletonDetector
from .hand_gesture_detector import Gesture, HandGestureDetector
from .mediapipe_detector import MediaPipeDetector
from .utils import (
    PerformanceTimer,
    calculate_angle,
    calculate_distance,
    create_output_directory,
    draw_fps_on_image,
    save_detection_summary,
    visualize_landmarks_3d,
)

__version__ = "0.1.0"

__all__ = [
    "BaseSkeletonDetector",
    "Gesture",
    "HandGestureDetector",
    "MediaPipeDetector",
    "PerformanceTimer",
    "calculate_angle",
    "calculate_distance",
    "create_output_directory",
    "draw_fps_on_image",
    "save_detection_summary",
    "visualize_landmarks_3d",
]
