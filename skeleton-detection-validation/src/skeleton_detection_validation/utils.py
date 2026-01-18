"""ユーティリティ関数"""

import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_angle(p1: dict[str, float], p2: dict[str, float], p3: dict[str, float]) -> float:
    """
    3点から角度を計算する (p2が頂点)

    Args:
        p1, p2, p3: ランドマーク座標 {'x': float, 'y': float, 'z': float}

    Returns:
        角度 (度数法)
    """
    # ベクトルを計算
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"], p1["z"] - p2["z"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"], p3["z"] - p2["z"]])

    # 内積とノルムから角度を計算
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # 数値誤差対策
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def calculate_distance(p1: dict[str, float], p2: dict[str, float]) -> float:
    """
    2点間の距離を計算する

    Args:
        p1, p2: ランドマーク座標 {'x': float, 'y': float, 'z': float}

    Returns:
        ユークリッド距離
    """
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    dz = p1["z"] - p2["z"]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


class PerformanceTimer:
    """処理時間を測定するためのクラス"""

    def __init__(self):
        self.times: list[float] = []
        self.start_time: float | None = None

    def start(self) -> None:
        """計測開始"""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """計測終了"""
        if self.start_time is None:
            raise RuntimeError("start()を先に呼び出してください")

        elapsed = time.perf_counter() - self.start_time
        self.times.append(elapsed)
        self.start_time = None
        return elapsed

    def get_average(self) -> float:
        """平均処理時間を取得"""
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)

    def get_fps(self) -> float:
        """平均FPSを取得"""
        avg = self.get_average()
        return 1.0 / avg if avg > 0 else 0.0

    def reset(self) -> None:
        """計測結果をリセット"""
        self.times = []
        self.start_time = None

    def get_stats(self) -> dict[str, float]:
        """統計情報を取得"""
        if not self.times:
            return {"count": 0, "average": 0.0, "min": 0.0, "max": 0.0, "fps": 0.0}

        return {
            "count": len(self.times),
            "average": self.get_average(),
            "min": min(self.times),
            "max": max(self.times),
            "fps": self.get_fps(),
        }


def visualize_landmarks_3d(landmarks: list[dict[str, float]], title: str = "3D Pose") -> None:
    """
    ランドマークを3Dプロットで可視化する

    Args:
        landmarks: ランドマーク座標のリスト
        title: グラフのタイトル
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ランドマークをプロット
    xs = [lm["x"] for lm in landmarks]
    ys = [lm["y"] for lm in landmarks]
    zs = [lm["z"] for lm in landmarks]

    ax.scatter(xs, ys, zs, c="red", marker="o")

    # ラベルを追加 (主要な点のみ)
    important_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    for i in important_indices:
        if i < len(landmarks):
            ax.text(xs[i], ys[i], zs[i], str(i), size=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plt.show()


def create_output_directory(base_dir: str | Path) -> Path:
    """
    出力ディレクトリを作成する

    Args:
        base_dir: ベースディレクトリ

    Returns:
        作成されたディレクトリのPath
    """
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_detection_summary(results: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    検出結果のサマリーを保存する

    Args:
        results: 検出結果のリスト
        output_path: 出力ファイルパス
    """
    total_frames = len(results)
    detected_frames = sum(1 for r in results if r["success"])
    detection_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0

    summary = f"""
=== 骨格検出サマリー ===
総フレーム数: {total_frames}
検出成功フレーム数: {detected_frames}
検出率: {detection_rate:.2f}%
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)


def draw_fps_on_image(image: np.ndarray, fps: float, position: tuple[int, int] = (10, 90)) -> np.ndarray:
    """
    画像にFPSを描画する

    Args:
        image: 入力画像
        fps: FPS値
        position: 描画位置 (x, y)

    Returns:
        FPSが描画された画像
    """
    text = f"FPS: {fps:.2f}"
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return image
