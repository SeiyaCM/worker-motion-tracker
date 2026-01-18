"""じゃんけんジェスチャー検出のサンプル"""

import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import cv2

from skeleton_detection_validation.hand_gesture_detector import Gesture, HandGestureDetector
from skeleton_detection_validation.utils import PerformanceTimer


def main():
    print("=" * 60)
    print("じゃんけんジェスチャー検出デモ (Webカメラ)")
    print("=" * 60)
    print("操作方法:")
    print("  - 'q'キー: 終了")
    print("=" * 60)
    print("\n検出可能なジェスチャー:")
    print("  - グー (Rock): 全ての指を握る")
    print("  - チョキ (Scissors): 人差し指と中指を伸ばす")
    print("  - パー (Paper): 全ての指を開く")
    print("=" * 60)
    print()

    # 検出器を初期化
    detector = HandGestureDetector(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # パフォーマンス測定
    timer = PerformanceTimer()

    # ジェスチャーカウンター
    gesture_counts = {
        Gesture.ROCK: 0,
        Gesture.SCISSORS: 0,
        Gesture.PAPER: 0,
        Gesture.UNKNOWN: 0,
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
        return

    try:
        print("カメラを起動しています...")

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

            # ランドマークとジェスチャーを描画
            output_frame = detector.draw_hand_landmarks(frame, result, show_gesture=True)

            # 検出されたジェスチャーをカウント
            if result["success"]:
                for hand in result["hands"]:
                    gesture = hand["gesture"]
                    gesture_counts[gesture] += 1

            elapsed = timer.stop()

            # FPS表示
            fps = 1.0 / elapsed if elapsed > 0 else 0
            cv2.putText(
                output_frame,
                f"FPS: {fps:.1f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            # 統計情報を表示
            y_offset = 120
            cv2.putText(
                output_frame,
                "Gesture Counts:",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            for gesture, count in gesture_counts.items():
                if gesture != Gesture.UNKNOWN:
                    y_offset += 20
                    label = f"  {gesture.value}: {count}"
                    cv2.putText(
                        output_frame,
                        label,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                    )

            # 操作方法を表示
            cv2.putText(
                output_frame,
                "Press 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Janken Gesture Detection", output_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n\nキーボード割り込みで終了しました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

        # 統計情報を表示
        stats = timer.get_stats()
        print("\n" + "=" * 60)
        print("統計情報:")
        print(f"  処理フレーム数: {stats['count']}")
        print(f"  平均処理時間: {stats['average']*1000:.2f}ms")
        print(f"  最小処理時間: {stats['min']*1000:.2f}ms")
        print(f"  最大処理時間: {stats['max']*1000:.2f}ms")
        print(f"  平均FPS: {stats['fps']:.2f}")
        print()
        print("検出されたジェスチャー:")
        for gesture, count in gesture_counts.items():
            if gesture != Gesture.UNKNOWN:
                print(f"  {gesture.value}: {count}回")
        print("=" * 60)


if __name__ == "__main__":
    main()
