"""Webカメラからリアルタイムで骨格検出を行うサンプル"""

import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from skeleton_detection_validation import MediaPipeDetector


def main():
    print("=" * 60)
    print("Webカメラ骨格検出デモ")
    print("=" * 60)
    print("操作方法:")
    print("  - 'q'キー: 終了")
    print("=" * 60)
    print()

    # 検出器を初期化
    detector = MediaPipeDetector(
        static_image_mode=False,  # 動画モード（トラッキング有効）
        model_complexity=1,  # バランス型モデル（リアルタイム処理向け）
        smooth_landmarks=True,  # ランドマークの平滑化を有効化
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        print("カメラを起動しています...")

        # Webカメラから骨格検出を実行
        # カメラID: 0 = デフォルトカメラ, 1 = 2台目のカメラ
        detector.process_webcam(camera_id=0)

        print("\n終了しました")

    except KeyboardInterrupt:
        print("\n\nキーボード割り込みで終了しました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
    finally:
        detector.close()


if __name__ == "__main__":
    main()
