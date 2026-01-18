"""MediaPipe Handsを使った手のジェスチャー検出の実装"""

from enum import Enum
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class Gesture(Enum):
    """じゃんけんジェスチャーの種類"""

    ROCK = "Rock"  # グー
    SCISSORS = "Scissors"  # チョキ
    PAPER = "Paper"  # パー
    UNKNOWN = "Unknown"  # 不明


class HandGestureDetector:
    """MediaPipe Handsを使った手のジェスチャー検出クラス"""

    # 指のランドマークインデックス
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2

    INDEX_FINGER_TIP = 8
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_MCP = 5

    MIDDLE_FINGER_TIP = 12
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_MCP = 9

    RING_FINGER_TIP = 16
    RING_FINGER_PIP = 14
    RING_FINGER_MCP = 13

    PINKY_TIP = 20
    PINKY_PIP = 18
    PINKY_MCP = 17

    WRIST = 0

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        HandGestureDetectorの初期化

        Args:
            static_image_mode: 静止画モード (Trueの場合、各画像を独立して処理)
            max_num_hands: 検出する手の最大数
            model_complexity: モデルの複雑さ (0, 1)
            min_detection_confidence: 検出の最小信頼度
            min_tracking_confidence: トラッキングの最小信頼度
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """
        画像から手のランドマークとジェスチャーを検出する

        Args:
            image: 入力画像 (BGR形式)

        Returns:
            検出結果を含む辞書
            {
                'success': 手が検出されたかどうか,
                'hands': [
                    {
                        'handedness': 'Left' or 'Right',
                        'landmarks': ランドマーク座標のリスト,
                        'gesture': Gesture enum
                    },
                    ...
                ],
                'raw_result': MediaPipeの生の結果
            }
        """
        # BGRからRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipeで処理
        results = self.hands.process(image_rgb)

        hands_data = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # ランドマークを辞書形式に変換
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append(
                        {
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                        }
                    )

                # ジェスチャーを認識
                gesture = self._recognize_gesture(hand_landmarks.landmark)

                # 左右の判定（MediaPipeは鏡像で返すため反転）
                hand_label = handedness.classification[0].label

                hands_data.append(
                    {
                        "handedness": hand_label,
                        "landmarks": landmarks,
                        "gesture": gesture,
                        "raw_landmarks": hand_landmarks,
                    }
                )

        return {
            "success": len(hands_data) > 0,
            "hands": hands_data,
            "raw_result": results,
        }

    def _is_finger_extended(
        self, landmarks: list, tip_idx: int, pip_idx: int, mcp_idx: int
    ) -> bool:
        """
        指が伸びているかどうかを判定する

        Args:
            landmarks: ランドマークのリスト
            tip_idx: 指先のインデックス
            pip_idx: PIP関節のインデックス
            mcp_idx: MCP関節のインデックス

        Returns:
            指が伸びていればTrue
        """
        # 指先がPIP関節より上（y座標が小さい）にあれば伸びている
        # また、指先がMCP関節より上にあることも確認
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]

        return tip.y < pip.y and tip.y < mcp.y

    def _is_thumb_extended(self, landmarks: list) -> bool:
        """
        親指が伸びているかどうかを判定する

        親指は他の指と異なり、横方向の判定も必要

        Args:
            landmarks: ランドマークのリスト

        Returns:
            親指が伸びていればTrue
        """
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        wrist = landmarks[self.WRIST]

        # 親指の先端がIP関節より外側にあるかを確認
        # 手首と親指MCPの位置関係で左右を判断
        if wrist.x < thumb_mcp.x:
            # 右手（画面上）: 親指が右に伸びる
            return thumb_tip.x > thumb_ip.x
        else:
            # 左手（画面上）: 親指が左に伸びる
            return thumb_tip.x < thumb_ip.x

    def _recognize_gesture(self, landmarks: list) -> Gesture:
        """
        ランドマークからじゃんけんのジェスチャーを認識する

        Args:
            landmarks: 手のランドマークのリスト

        Returns:
            認識されたジェスチャー
        """
        # 各指の状態を判定
        thumb_extended = self._is_thumb_extended(landmarks)
        index_extended = self._is_finger_extended(
            landmarks, self.INDEX_FINGER_TIP, self.INDEX_FINGER_PIP, self.INDEX_FINGER_MCP
        )
        middle_extended = self._is_finger_extended(
            landmarks, self.MIDDLE_FINGER_TIP, self.MIDDLE_FINGER_PIP, self.MIDDLE_FINGER_MCP
        )
        ring_extended = self._is_finger_extended(
            landmarks, self.RING_FINGER_TIP, self.RING_FINGER_PIP, self.RING_FINGER_MCP
        )
        pinky_extended = self._is_finger_extended(
            landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP
        )

        # 伸びている指の数をカウント
        extended_count = sum(
            [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        )

        # ジェスチャー判定
        # パー: 全ての指が伸びている (4-5本)
        if extended_count >= 4:
            return Gesture.PAPER

        # グー: 全ての指が曲がっている (0-1本)
        if extended_count <= 1:
            return Gesture.ROCK

        # チョキ: 人差し指と中指が伸びていて、薬指と小指が曲がっている
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return Gesture.SCISSORS

        return Gesture.UNKNOWN

    def draw_hand_landmarks(
        self, image: np.ndarray, result: dict[str, Any], show_gesture: bool = True
    ) -> np.ndarray:
        """
        検出結果を画像に描画する

        Args:
            image: 入力画像 (BGR形式)
            result: detect()メソッドの返り値
            show_gesture: ジェスチャー名を表示するかどうか

        Returns:
            ランドマークが描画された画像
        """
        output_image = image.copy()

        if result["success"]:
            for i, hand in enumerate(result["hands"]):
                # ランドマークを描画
                self.mp_drawing.draw_landmarks(
                    output_image,
                    hand["raw_landmarks"],
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                if show_gesture:
                    # ジェスチャー名を表示
                    handedness = hand["handedness"]
                    gesture = hand["gesture"]

                    # 表示位置を計算（手首の位置を基準に）
                    wrist = hand["landmarks"][self.WRIST]
                    h, w = image.shape[:2]
                    x = int(wrist["x"] * w)
                    y = int(wrist["y"] * h) - 20

                    # ジェスチャーに応じた色
                    color = self._get_gesture_color(gesture)

                    # 日本語ラベル
                    gesture_label = self._get_gesture_label(gesture)

                    text = f"{handedness}: {gesture_label}"
                    cv2.putText(
                        output_image,
                        text,
                        (x - 50, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

            # 検出成功を表示
            cv2.putText(
                output_image,
                f"Hands Detected: {len(result['hands'])}",
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
                "No Hands Detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return output_image

    def _get_gesture_color(self, gesture: Gesture) -> tuple[int, int, int]:
        """ジェスチャーに応じた色を返す"""
        colors = {
            Gesture.ROCK: (0, 0, 255),  # 赤
            Gesture.SCISSORS: (0, 255, 255),  # 黄
            Gesture.PAPER: (0, 255, 0),  # 緑
            Gesture.UNKNOWN: (128, 128, 128),  # グレー
        }
        return colors.get(gesture, (255, 255, 255))

    def _get_gesture_label(self, gesture: Gesture) -> str:
        """ジェスチャーの日本語ラベルを返す"""
        labels = {
            Gesture.ROCK: "Rock (Guu)",
            Gesture.SCISSORS: "Scissors (Choki)",
            Gesture.PAPER: "Paper (Paa)",
            Gesture.UNKNOWN: "Unknown",
        }
        return labels.get(gesture, "Unknown")

    def get_landmark_names(self) -> list[str]:
        """
        ランドマークの名前リストを取得する

        Returns:
            ランドマーク名のリスト (21個)
        """
        return [landmark.name for landmark in self.mp_hands.HandLandmark]

    def close(self) -> None:
        """リソースを解放する"""
        self.hands.close()
