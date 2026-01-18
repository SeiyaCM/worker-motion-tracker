"""骨格検出の基底クラス"""

from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np


class BaseSkeletonDetector(ABC):
    """骨格検出の基底クラス"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """
        画像から骨格を検出する

        Args:
            image: 入力画像 (BGR形式)

        Returns:
            検出結果を含む辞書
            {
                'landmarks': 検出されたランドマーク座標のリスト,
                'success': 検出成功フラグ,
                'confidence': 信頼度 (オプション)
            }
        """
        pass

    @abstractmethod
    def draw_skeleton(self, image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
        """
        検出結果を画像に描画する

        Args:
            image: 入力画像 (BGR形式)
            result: detect()メソッドの返り値

        Returns:
            骨格が描画された画像
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """リソースを解放する"""
        pass

    def process_image_file(self, image_path: str, output_path: str | None = None) -> dict[str, Any]:
        """
        画像ファイルから骨格検出を実行する

        Args:
            image_path: 入力画像のパス
            output_path: 出力画像のパス (Noneの場合は保存しない)

        Returns:
            検出結果
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        result = self.detect(image)

        if output_path is not None:
            output_image = self.draw_skeleton(image.copy(), result)
            cv2.imwrite(output_path, output_image)

        return result

    def process_video_file(
        self, video_path: str, output_path: str | None = None, display: bool = False
    ) -> list[dict[str, Any]]:
        """
        動画ファイルから骨格検出を実行する

        Args:
            video_path: 入力動画のパス
            output_path: 出力動画のパス (Noneの場合は保存しない)
            display: リアルタイムで表示するかどうか

        Returns:
            各フレームの検出結果のリスト
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        # 動画情報を取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 出力動画の設定
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.detect(frame)
                results.append(result)

                output_frame = self.draw_skeleton(frame, result)

                if writer is not None:
                    writer.write(output_frame)

                if display:
                    cv2.imshow("Skeleton Detection", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        return results

    def process_webcam(self, camera_id: int = 0) -> None:
        """
        Webカメラからリアルタイムで骨格検出を実行する

        Args:
            camera_id: カメラID (デフォルト: 0)
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError("カメラを開けませんでした")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.detect(frame)
                output_frame = self.draw_skeleton(frame, result)

                # FPS表示
                cv2.putText(
                    output_frame,
                    "Press 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Webcam Skeleton Detection", output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
