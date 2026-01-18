"""関節角度を測定するサンプル"""

import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import cv2

from skeleton_detection_validation import MediaPipeDetector, PerformanceTimer, calculate_angle


def main():
    print("=" * 60)
    print("関節角度測定デモ (Webカメラ)")
    print("=" * 60)
    print("操作方法:")
    print("  - 'q'キー: 終了")
    print("=" * 60)
    print("\n測定する関節角度:")
    print("  - 左肘の角度 (左肩-左肘-左手首)")
    print("  - 右肘の角度 (右肩-右肘-右手首)")
    print("  - 左膝の角度 (左腰-左膝-左足首)")
    print("  - 右膝の角度 (右腰-右膝-右足首)")
    print("=" * 60)
    print()

    # 検出器を初期化
    detector = MediaPipeDetector(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # パフォーマンス測定
    timer = PerformanceTimer()

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

            # 処理時間を測定
            timer.start()

            # 骨格検出
            result = detector.detect(frame)

            # 骨格を描画
            output_frame = detector.draw_skeleton(frame, result)

            # 角度を計算して描画
            if result["success"]:
                # 左肘の角度
                left_shoulder = detector.get_landmark_by_name(result, "LEFT_SHOULDER")
                left_elbow = detector.get_landmark_by_name(result, "LEFT_ELBOW")
                left_wrist = detector.get_landmark_by_name(result, "LEFT_WRIST")

                if left_shoulder and left_elbow and left_wrist:
                    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    cv2.putText(
                        output_frame,
                        f"Left Elbow: {angle:.1f}deg",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                # 右肘の角度
                right_shoulder = detector.get_landmark_by_name(result, "RIGHT_SHOULDER")
                right_elbow = detector.get_landmark_by_name(result, "RIGHT_ELBOW")
                right_wrist = detector.get_landmark_by_name(result, "RIGHT_WRIST")

                if right_shoulder and right_elbow and right_wrist:
                    angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    cv2.putText(
                        output_frame,
                        f"Right Elbow: {angle:.1f}deg",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                # 左膝の角度
                left_hip = detector.get_landmark_by_name(result, "LEFT_HIP")
                left_knee = detector.get_landmark_by_name(result, "LEFT_KNEE")
                left_ankle = detector.get_landmark_by_name(result, "LEFT_ANKLE")

                if left_hip and left_knee and left_ankle:
                    angle = calculate_angle(left_hip, left_knee, left_ankle)
                    cv2.putText(
                        output_frame,
                        f"Left Knee: {angle:.1f}deg",
                        (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                # 右膝の角度
                right_hip = detector.get_landmark_by_name(result, "RIGHT_HIP")
                right_knee = detector.get_landmark_by_name(result, "RIGHT_KNEE")
                right_ankle = detector.get_landmark_by_name(result, "RIGHT_ANKLE")

                if right_hip and right_knee and right_ankle:
                    angle = calculate_angle(right_hip, right_knee, right_ankle)
                    cv2.putText(
                        output_frame,
                        f"Right Knee: {angle:.1f}deg",
                        (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

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

            cv2.imshow("Joint Angle Measurement", output_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n\nキーボード割り込みで終了しました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
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
        print("=" * 60)


if __name__ == "__main__":
    main()
