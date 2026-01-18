"""ユーティリティ関数"""

import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from skeleton_detection_validation.hand_gesture_detector import Gesture


class CycleState(Enum):
    """ジェスチャーサイクルの状態"""

    WAITING_ROCK = "waiting_rock"  # グーを待っている
    WAITING_SCISSORS = "waiting_scissors"  # チョキを待っている
    WAITING_PAPER = "waiting_paper"  # パーを待っている
    CYCLE_COMPLETE = "cycle_complete"  # サイクル完了


class GestureCycleTracker:
    """
    グー→チョキ→パーのジェスチャーサイクルを追跡するクラス

    状態遷移:
    WAITING_ROCK → WAITING_SCISSORS → WAITING_PAPER → CYCLE_COMPLETE → WAITING_ROCK ...

    安定化:
    同じジェスチャーが連続で指定回数（デフォルト5回）検出されたら確定
    """

    # ジェスチャー名の定数（Gesture enumとの循環インポートを避けるため）
    GESTURE_ROCK = "Rock"
    GESTURE_SCISSORS = "Scissors"
    GESTURE_PAPER = "Paper"

    def __init__(self, max_cycles: int = 3, stability_threshold: int = 5):
        """
        GestureCycleTrackerの初期化

        Args:
            max_cycles: 完了までのサイクル数（デフォルト3）
            stability_threshold: ジェスチャー確定に必要な連続フレーム数（デフォルト5）
        """
        self.max_cycles = max_cycles
        self.stability_threshold = stability_threshold

        # 状態
        self._state = CycleState.WAITING_ROCK
        self._current_cycle = 0
        self._lap_times: list[float] = []

        # タイミング
        self._cycle_start_time: float | None = None
        self._total_start_time: float | None = None

        # 安定化用
        self._last_gesture: str | None = None
        self._consecutive_count = 0

    def update(self, gesture: "Gesture") -> dict[str, Any]:
        """
        現在のジェスチャーで状態を更新する

        Args:
            gesture: 検出されたジェスチャー（Gesture enum）

        Returns:
            更新結果を含む辞書:
            {
                'state_changed': 状態が変わったかどうか,
                'cycle_completed': サイクルが完了したかどうか,
                'all_completed': 全サイクルが完了したかどうか,
                'current_lap_time': 現在のサイクルの経過時間（秒）,
                'message': 状態に関するメッセージ
            }
        """
        gesture_value = gesture.value

        result = {
            "state_changed": False,
            "cycle_completed": False,
            "all_completed": False,
            "current_lap_time": self._get_current_lap_time(),
            "message": "",
        }

        # 全サイクル完了済みの場合
        if self.is_complete():
            result["all_completed"] = True
            result["message"] = "All cycles completed!"
            return result

        # 安定化ロジック: 連続フレームをカウント
        if gesture_value == self._last_gesture:
            self._consecutive_count += 1
        else:
            self._last_gesture = gesture_value
            self._consecutive_count = 1

        # 安定化閾値に達していない場合は状態変更しない
        if self._consecutive_count < self.stability_threshold:
            return result

        # 期待するジェスチャーと一致するか確認
        expected = self._get_expected_gesture_value()
        if gesture_value != expected:
            return result

        # 状態遷移を実行
        result["state_changed"] = True

        if self._state == CycleState.WAITING_ROCK:
            # サイクル開始
            self._cycle_start_time = time.perf_counter()
            if self._total_start_time is None:
                self._total_start_time = self._cycle_start_time
            self._state = CycleState.WAITING_SCISSORS
            result["message"] = "Rock detected! Now show Scissors."

        elif self._state == CycleState.WAITING_SCISSORS:
            self._state = CycleState.WAITING_PAPER
            result["message"] = "Scissors detected! Now show Paper."

        elif self._state == CycleState.WAITING_PAPER:
            # サイクル完了
            lap_time = time.perf_counter() - self._cycle_start_time
            self._lap_times.append(lap_time)
            self._current_cycle += 1
            result["cycle_completed"] = True
            result["message"] = f"Cycle {self._current_cycle} complete! Lap time: {lap_time:.2f}s"

            if self._current_cycle >= self.max_cycles:
                self._state = CycleState.CYCLE_COMPLETE
                result["all_completed"] = True
                result["message"] = f"All {self.max_cycles} cycles completed!"
            else:
                self._state = CycleState.WAITING_ROCK
                result["message"] += " Start next cycle with Rock."

        # 安定化カウンターをリセット（状態が変わったので）
        self._consecutive_count = 0
        self._last_gesture = None

        return result

    def get_expected_gesture(self) -> str:
        """
        次に期待するジェスチャーの表示名を取得する

        Returns:
            期待するジェスチャーの表示名（日本語付き）
        """
        gesture_labels = {
            self.GESTURE_ROCK: "Rock (Guu)",
            self.GESTURE_SCISSORS: "Scissors (Choki)",
            self.GESTURE_PAPER: "Paper (Paa)",
        }
        expected = self._get_expected_gesture_value()
        return gesture_labels.get(expected, "Complete")

    def _get_expected_gesture_value(self) -> str | None:
        """
        次に期待するジェスチャーの値を取得する

        Returns:
            期待するジェスチャーの値（Gesture.valueと同じ形式）
        """
        if self._state == CycleState.WAITING_ROCK:
            return self.GESTURE_ROCK
        elif self._state == CycleState.WAITING_SCISSORS:
            return self.GESTURE_SCISSORS
        elif self._state == CycleState.WAITING_PAPER:
            return self.GESTURE_PAPER
        return None

    def get_lap_times(self) -> list[float]:
        """
        完了したサイクルのラップタイムを取得する

        Returns:
            各サイクルの所要時間（秒）のリスト
        """
        return self._lap_times.copy()

    def get_current_cycle(self) -> int:
        """
        現在のサイクル番号を取得する（1から始まる）

        Returns:
            現在のサイクル番号（進行中のサイクル）
        """
        if self.is_complete():
            return self.max_cycles
        return self._current_cycle + 1

    def _get_current_lap_time(self) -> float:
        """
        現在のサイクルの経過時間を取得する

        Returns:
            現在のサイクルの経過時間（秒）。サイクル未開始の場合は0.0
        """
        if self._cycle_start_time is None:
            return 0.0
        return time.perf_counter() - self._cycle_start_time

    def get_total_time(self) -> float:
        """
        測定開始からの合計時間を取得する

        Returns:
            合計時間（秒）。測定未開始の場合は0.0
        """
        if self._total_start_time is None:
            return 0.0
        return time.perf_counter() - self._total_start_time

    def get_average_lap_time(self) -> float:
        """
        平均ラップタイムを取得する

        Returns:
            平均ラップタイム（秒）。ラップがない場合は0.0
        """
        if not self._lap_times:
            return 0.0
        return sum(self._lap_times) / len(self._lap_times)

    def is_complete(self) -> bool:
        """
        全サイクルが完了したかどうかを確認する

        Returns:
            全サイクル完了ならTrue
        """
        return self._current_cycle >= self.max_cycles

    def is_started(self) -> bool:
        """
        測定が開始されているかどうかを確認する

        Returns:
            測定開始済みならTrue
        """
        return self._cycle_start_time is not None

    def get_state(self) -> CycleState:
        """
        現在の状態を取得する

        Returns:
            現在のCycleState
        """
        return self._state

    def get_stability_progress(self) -> tuple[int, int]:
        """
        安定化の進捗を取得する

        Returns:
            (現在の連続カウント, 必要な閾値)のタプル
        """
        return (self._consecutive_count, self.stability_threshold)

    def reset(self) -> None:
        """状態をリセットする"""
        self._state = CycleState.WAITING_ROCK
        self._current_cycle = 0
        self._lap_times = []
        self._cycle_start_time = None
        self._total_start_time = None
        self._last_gesture = None
        self._consecutive_count = 0


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
