"""動画ファイルから骨格検出を行うサンプル"""

import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from skeleton_detection_validation import (
    MediaPipeDetector,
    PerformanceTimer,
    create_output_directory,
    save_detection_summary,
)


def main():
    # 出力ディレクトリを作成
    output_dir = create_output_directory(project_root / "results" / "videos")

    # 検出器を初期化
    detector = MediaPipeDetector(
        static_image_mode=False,  # 動画モード（トラッキング有効）
        model_complexity=1,  # バランス型モデル
        smooth_landmarks=True,  # ランドマークの平滑化を有効化
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # テスト動画のパス（実際のパスに置き換えてください）
    video_path = project_root / "data" / "videos" / "test_video.mp4"
    output_path = output_dir / "result_video.mp4"
    summary_path = output_dir / "summary.txt"

    try:
        # 動画が存在するか確認
        if not video_path.exists():
            print(f"エラー: 動画ファイルが見つかりません: {video_path}")
            print(f"data/videos/ フォルダにテスト動画を配置してください")
            return

        print(f"処理中: {video_path}")
        print("処理には時間がかかる場合があります...")
        print("リアルタイムプレビューを表示します（'q'キーで中断）\n")

        # パフォーマンス測定
        timer = PerformanceTimer()
        timer.start()

        # 骨格検出を実行
        results = detector.process_video_file(
            str(video_path), str(output_path), display=True  # リアルタイム表示を有効化
        )

        elapsed = timer.stop()

        # 統計情報を計算
        total_frames = len(results)
        detected_frames = sum(1 for r in results if r["success"])
        detection_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0

        # 結果を表示
        print("\n" + "=" * 60)
        print("処理完了！")
        print("=" * 60)
        print(f"総フレーム数: {total_frames}")
        print(f"検出成功フレーム数: {detected_frames}")
        print(f"検出率: {detection_rate:.2f}%")
        print(f"処理時間: {elapsed:.2f}秒")
        print(f"平均FPS: {total_frames / elapsed:.2f}")
        print(f"\n出力動画: {output_path}")
        print(f"サマリー: {summary_path}")
        print("=" * 60)

        # サマリーを保存
        save_detection_summary(results, summary_path)

    finally:
        detector.close()


if __name__ == "__main__":
    main()
