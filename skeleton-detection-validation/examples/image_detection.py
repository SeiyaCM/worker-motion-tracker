"""静止画から骨格検出を行うサンプル"""

import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from skeleton_detection_validation import MediaPipeDetector, create_output_directory


def main():
    # 出力ディレクトリを作成
    output_dir = create_output_directory(project_root / "results" / "images")

    # 検出器を初期化
    detector = MediaPipeDetector(
        static_image_mode=True,  # 静止画モード
        model_complexity=2,  # 高精度モデル
        min_detection_confidence=0.5,
    )

    # テスト画像のパス（実際のパスに置き換えてください）
    image_path = project_root / "data" / "images" / "test_image.jpg"
    output_path = output_dir / "result_image.jpg"

    try:
        # 画像が存在するか確認
        if not image_path.exists():
            print(f"エラー: 画像ファイルが見つかりません: {image_path}")
            print(f"data/images/ フォルダにテスト画像を配置してください")
            return

        print(f"処理中: {image_path}")

        # 骨格検出を実行
        result = detector.process_image_file(str(image_path), str(output_path))

        # 結果を表示
        if result["success"]:
            print(f"✓ 骨格検出成功！")
            print(f"  検出されたランドマーク数: {len(result['landmarks'])}")
            print(f"  結果画像: {output_path}")

            # 主要なランドマーク座標を表示
            print("\n主要なランドマーク:")
            important_landmarks = [
                "NOSE",
                "LEFT_SHOULDER",
                "RIGHT_SHOULDER",
                "LEFT_ELBOW",
                "RIGHT_ELBOW",
                "LEFT_WRIST",
                "RIGHT_WRIST",
                "LEFT_HIP",
                "RIGHT_HIP",
            ]

            for name in important_landmarks:
                landmark = detector.get_landmark_by_name(result, name)
                if landmark:
                    print(
                        f"  {name}: "
                        f"x={landmark['x']:.3f}, "
                        f"y={landmark['y']:.3f}, "
                        f"visibility={landmark['visibility']:.3f}"
                    )
        else:
            print("✗ 骨格が検出できませんでした")

    finally:
        detector.close()


if __name__ == "__main__":
    main()
