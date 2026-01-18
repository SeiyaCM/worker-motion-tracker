"""OpenCV and MediaPipe skeleton detection validation package."""

from .base_detector import BaseSkeletonDetector
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
    "MediaPipeDetector",
    "PerformanceTimer",
    "calculate_angle",
    "calculate_distance",
    "create_output_directory",
    "draw_fps_on_image",
    "save_detection_summary",
    "visualize_landmarks_3d",
]
