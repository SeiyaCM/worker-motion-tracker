"""グー→チョキ→パーのジェスチャーサイクル測定サンプル

Webカメラを使ってグー→チョキ→パーのサイクルタイムを測定します。
3サイクル完了で自動終了し、各ラップタイムと統計情報を表示します。
"""

import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import cv2

from skeleton_detection_validation.hand_gesture_detector import Gesture, HandGestureDetector
from skeleton_detection_validation.utils import GestureCycleTracker, PerformanceTimer


def draw_info_panel(
    image, cycle_tracker: GestureCycleTracker, detected_gesture: Gesture | None, fps: float
) -> None:
    """
    情報パネルを画像に描画する

    Args:
        image: 描画対象の画像
        cycle_tracker: サイクルトラッカー
        detected_gesture: 検出されたジェスチャー
        fps: 現在のFPS
    """
    h, w = image.shape[:2]

    # 半透明の背景パネルを描画
    panel_height = 200
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # テキストの共通設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    cyan = (255, 255, 0)

    y_offset = 25
    line_height = 28

    # サイクル情報
    current_cycle = cycle_tracker.get_current_cycle()
    max_cycles = cycle_tracker.max_cycles
    cv2.putText(
        image,
        f"Cycle: {current_cycle}/{max_cycles}",
        (10, y_offset),
        font,
        0.8,
        yellow,
        2,
    )
    y_offset += line_height

    # 次に期待するジェスチャー
    expected = cycle_tracker.get_expected_gesture()
    if cycle_tracker.is_complete():
        cv2.putText(
            image,
            "COMPLETE!",
            (10, y_offset),
            font,
            0.8,
            green,
            2,
        )
    else:
        cv2.putText(
            image,
            f"Next: {expected}",
            (10, y_offset),
            font,
            0.7,
            white,
            2,
        )
    y_offset += line_height

    # 安定化プログレス
    if not cycle_tracker.is_complete():
        progress, threshold = cycle_tracker.get_stability_progress()
        if progress > 0:
            cv2.putText(
                image,
                f"Stability: {progress}/{threshold}",
                (10, y_offset),
                font,
                0.5,
                cyan,
                1,
            )
    y_offset += line_height - 5

    # 現在のサイクル経過時間
    if cycle_tracker.is_started() and not cycle_tracker.is_complete():
        current_lap = cycle_tracker._get_current_lap_time()
        cv2.putText(
            image,
            f"Current: {current_lap:.2f}s",
            (10, y_offset),
            font,
            0.6,
            white,
            1,
        )
    y_offset += line_height - 5

    # 区切り線
    cv2.line(image, (10, y_offset - 5), (340, y_offset - 5), (100, 100, 100), 1)

    # ラップタイム一覧
    lap_times = cycle_tracker.get_lap_times()
    for i, lap_time in enumerate(lap_times, start=1):
        cv2.putText(
            image,
            f"Lap {i}: {lap_time:.2f}s",
            (10, y_offset),
            font,
            0.5,
            green,
            1,
        )
        y_offset += 20

    # FPS表示（右上）
    cv2.putText(
        image,
        f"FPS: {fps:.1f}",
        (w - 120, 30),
        font,
        0.6,
        (255, 255, 0),
        2,
    )

    # 現在検出されているジェスチャー（右下）
    if detected_gesture:
        gesture_text = detected_gesture.value
        color = {
            Gesture.ROCK: (0, 0, 255),
            Gesture.SCISSORS: (0, 255, 255),
            Gesture.PAPER: (0, 255, 0),
            Gesture.UNKNOWN: (128, 128, 128),
        }.get(detected_gesture, (255, 255, 255))

        cv2.putText(
            image,
            f"Detected: {gesture_text}",
            (w - 200, h - 30),
            font,
            0.7,
            color,
            2,
        )


def draw_completion_overlay(image, cycle_tracker: GestureCycleTracker) -> None:
    """
    完了時のオーバーレイを描画する

    Args:
        image: 描画対象の画像
        cycle_tracker: サイクルトラッカー
    """
    h, w = image.shape[:2]

    # 半透明の背景
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 100, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    yellow = (0, 255, 255)

    # タイトル
    cv2.putText(
        image,
        "ALL CYCLES COMPLETE!",
        (w // 2 - 200, h // 2 - 80),
        font,
        1.0,
        yellow,
        3,
    )

    # ラップタイム
    lap_times = cycle_tracker.get_lap_times()
    y_offset = h // 2 - 30
    for i, lap_time in enumerate(lap_times, start=1):
        cv2.putText(
            image,
            f"Lap {i}: {lap_time:.2f}s",
            (w // 2 - 80, y_offset),
            font,
            0.8,
            white,
            2,
        )
        y_offset += 35

    # 平均時間
    avg_time = cycle_tracker.get_average_lap_time()
    cv2.putText(
        image,
        f"Average: {avg_time:.2f}s",
        (w // 2 - 100, y_offset + 20),
        font,
        0.9,
        yellow,
        2,
    )

    # 終了案内
    cv2.putText(
        image,
        "Press 'q' to quit, 'r' to restart",
        (w // 2 - 180, h - 50),
        font,
        0.7,
        white,
        2,
    )


def main():
    print("=" * 60)
    print("ジェスチャーサイクル測定")
    print("=" * 60)
    print("測定方法:")
    print("  1. グー (Rock) を出す")
    print("  2. チョキ (Scissors) を出す")
    print("  3. パー (Paper) を出す")
    print("  これを3サイクル繰り返します。")
    print()
    print("操作方法:")
    print("  - 'q'キー: 終了")
    print("  - 'r'キー: リスタート")
    print("=" * 60)
    print()

    # 検出器を初期化
    detector = HandGestureDetector(
        static_image_mode=False,
        max_num_hands=1,  # 1つの手のみ検出
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # サイクルトラッカーを初期化
    cycle_tracker = GestureCycleTracker(max_cycles=3, stability_threshold=5)

    # パフォーマンス測定
    timer = PerformanceTimer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
        return

    try:
        print("カメラを起動しています...")
        print("グー (Rock) を出して測定を開始してください。\n")

        completed_and_shown = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 鏡像に反転（より直感的な操作のため）
            frame = cv2.flip(frame, 1)

            # 処理時間を測定
            timer.start()

            # ジェスチャー検出
            result = detector.detect(frame)

            # ランドマークを描画
            output_frame = detector.draw_hand_landmarks(frame, result, show_gesture=False)

            # 検出されたジェスチャーを取得
            detected_gesture = None
            if result["success"] and result["hands"]:
                detected_gesture = result["hands"][0]["gesture"]

                # サイクルトラッカーを更新
                update_result = cycle_tracker.update(detected_gesture)

                # 状態変化があった場合はコンソールに出力
                if update_result["state_changed"]:
                    print(update_result["message"])

            elapsed = timer.stop()
            fps = 1.0 / elapsed if elapsed > 0 else 0

            # 情報パネルを描画
            draw_info_panel(output_frame, cycle_tracker, detected_gesture, fps)

            # 完了時のオーバーレイ
            if cycle_tracker.is_complete():
                if not completed_and_shown:
                    # 統計情報をコンソールに出力（1回のみ）
                    print("\n" + "=" * 60)
                    print("測定完了!")
                    print("=" * 60)
                    lap_times = cycle_tracker.get_lap_times()
                    for i, lap_time in enumerate(lap_times, start=1):
                        print(f"  Lap {i}: {lap_time:.2f}s")
                    print(f"\n  平均: {cycle_tracker.get_average_lap_time():.2f}s")
                    print(f"  合計: {sum(lap_times):.2f}s")
                    print("=" * 60)
                    print("'q'キーで終了、'r'キーでリスタート")
                    completed_and_shown = True

                draw_completion_overlay(output_frame, cycle_tracker)

            cv2.imshow("Gesture Cycle Measurement", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # リスタート
                cycle_tracker.reset()
                timer.reset()
                completed_and_shown = False
                print("\n" + "=" * 60)
                print("測定をリスタートしました。")
                print("グー (Rock) を出して測定を開始してください。")
                print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\nキーボード割り込みで終了しました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

        # 最終統計情報
        if cycle_tracker.get_lap_times():
            print("\n" + "=" * 60)
            print("最終統計情報:")
            print("=" * 60)
            lap_times = cycle_tracker.get_lap_times()
            for i, lap_time in enumerate(lap_times, start=1):
                print(f"  Lap {i}: {lap_time:.2f}s")
            if lap_times:
                print(f"\n  平均ラップタイム: {cycle_tracker.get_average_lap_time():.2f}s")
                print(f"  最速ラップ: {min(lap_times):.2f}s")
                print(f"  最遅ラップ: {max(lap_times):.2f}s")
            print("=" * 60)


if __name__ == "__main__":
    main()
