"""MediaPipeを使った骨格検出の実装"""

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .base_detector import BaseSkeletonDetector


class MediaPipeDetector(BaseSkeletonDetector):
    """MediaPipeを使った骨格検出クラス"""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        MediaPipeDetectorの初期化

        Args:
            static_image_mode: 静止画モード (Trueの場合、各画像を独立して処理)
            model_complexity: モデルの複雑さ (0, 1, 2)。高いほど精度が上がるが遅くなる
            smooth_landmarks: ランドマークの平滑化を行うか
            min_detection_confidence: 検出の最小信頼度
            min_tracking_confidence: トラッキングの最小信頼度
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """
        画像から骨格を検出する

        Args:
            image: 入力画像 (BGR形式)

        Returns:
            検出結果を含む辞書
        """
        # BGRからRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipeで処理
        results = self.pose.process(image_rgb)

        # 結果を整形
        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.append(
                    {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                    }
                )

        return {
            "success": results.pose_landmarks is not None,
            "landmarks": landmarks,
            "raw_result": results,
        }

    def draw_skeleton(self, image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
        """
        検出結果を画像に描画する

        Args:
            image: 入力画像 (BGR形式)
            result: detect()メソッドの返り値

        Returns:
            骨格が描画された画像
        """
        output_image = image.copy()

        if result["success"] and result["raw_result"].pose_landmarks:
            # MediaPipeの描画ユーティリティを使用
            self.mp_drawing.draw_landmarks(
                output_image,
                result["raw_result"].pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            # 検出成功を表示
            cv2.putText(
                output_image,
                "Pose Detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            # 検出失敗を表示
            cv2.putText(
                output_image,
                "No Pose Detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return output_image

    def get_landmark_names(self) -> list[str]:
        """
        ランドマークの名前リストを取得する

        Returns:
            ランドマーク名のリスト (33個)
        """
        return [landmark.name for landmark in self.mp_pose.PoseLandmark]

    def get_landmark_by_name(self, result: dict[str, Any], name: str) -> dict[str, float] | None:
        """
        名前を指定してランドマークを取得する

        Args:
            result: detect()メソッドの返り値
            name: ランドマーク名 (例: 'LEFT_SHOULDER', 'RIGHT_ELBOW')

        Returns:
            ランドマークの座標辞書、見つからない場合はNone
        """
        if not result["success"]:
            return None

        try:
            landmark_index = self.mp_pose.PoseLandmark[name].value
            return result["landmarks"][landmark_index]
        except (KeyError, IndexError):
            return None

    def close(self) -> None:
        """リソースを解放する"""
        self.pose.close()
