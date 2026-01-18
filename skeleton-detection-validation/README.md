# 骨格検出検証プロジェクト (Skeleton Detection Validation)

OpenCVとMediaPipeを使用した骨格検出の技術検証プロジェクトです。

## 概要

このプロジェクトは、作業者の動作追跡とサイクルタイム測定のための技術検証として、MediaPipeを使用した骨格検出の精度とパフォーマンスを評価します。

### 主な機能

- **静止画からの骨格検出**: 画像ファイルから33点のランドマークを検出
- **動画ファイルからの骨格検出**: 動画全体を処理して結果を保存
- **Webカメラからのリアルタイム検出**: リアルタイムで骨格を検出・表示
- **関節角度の測定**: 肘、膝などの関節角度を計算
- **パフォーマンス測定**: 処理速度（FPS）や検出率を計測

## 必要要件

- **Python**: 3.12 (3.13はMediaPipeが未対応)
- **OS**: Windows, macOS, Linux
- **カメラ**: Webカメラ検出を行う場合

## セットアップ

### 1. Python 3.12の確認

```powershell
# Pythonバージョンの確認
python --version  # Python 3.12.x であることを確認
```

Python 3.12がインストールされていない場合は、[python.org](https://www.python.org/downloads/)からインストールしてください。

### 2. 仮想環境の作成と有効化

プロジェクトルート（worker-motion-tracker）で実行：

```powershell
# 仮想環境を作成（まだ作成していない場合）
python -m venv .venv

# 仮想環境を有効化
.\.venv\Scripts\Activate.ps1
```

実行ポリシーエラーが出た場合：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. skeleton-detection-validationフォルダに移動

```powershell
cd skeleton-detection-validation
```

### 4. 依存パッケージのインストール

```powershell
# uvを使用してパッケージをインストール
uv pip install -e .
```

インストールされるパッケージ:
- `opencv-python>=4.10.0`: 画像・動画処理
- `mediapipe==0.10.9`: Google製の骨格検出ライブラリ（安定版）
- `numpy>=1.24.0`: 数値計算
- `matplotlib>=3.8.0`: 可視化
- `pillow>=10.0.0`: 画像処理補助

## プロジェクト構成

```
skeleton-detection-validation/
├── src/
│   └── skeleton_detection_validation/
│       ├── __init__.py                # パッケージ初期化
│       ├── base_detector.py           # 骨格検出の基底クラス
│       ├── mediapipe_detector.py      # MediaPipe実装
│       └── utils.py                   # ユーティリティ関数
├── examples/
│   ├── image_detection.py             # 静止画検出サンプル
│   ├── video_detection.py             # 動画検出サンプル
│   ├── webcam_detection.py            # Webカメラ検出サンプル
│   └── angle_measurement.py           # 関節角度測定サンプル
├── data/
│   ├── images/                        # テスト画像用フォルダ
│   └── videos/                        # テスト動画用フォルダ
├── results/
│   ├── images/                        # 処理結果画像
│   ├── videos/                        # 処理結果動画
│   └── reports/                       # 検証レポート
├── pyproject.toml                     # プロジェクト設定
└── README.md                          # このファイル
```

## 使い方

### 1. Webカメラでリアルタイム検出（推奨：最初のテスト）

```powershell
python examples/webcam_detection.py
```

- カメラの前に立つと骨格が自動検出されます
- `q`キーで終了

### 2. 静止画から骨格検出

```powershell
# data/images/ にテスト画像（test_image.jpg）を配置してから実行
python examples/image_detection.py
```

結果は `results/images/result_image.jpg` に保存されます。

### 3. 動画ファイルから骨格検出

```powershell
# data/videos/ にテスト動画（test_video.mp4）を配置してから実行
python examples/video_detection.py
```

結果は `results/videos/result_video.mp4` に保存されます。

### 4. 関節角度の測定

```powershell
python examples/angle_measurement.py
```

Webカメラから肘や膝の角度をリアルタイムで測定します。

## MediaPipeについて

### 検出可能なランドマーク（33点）

MediaPipeは以下の主要なランドマークを検出します：

- **顔部**: 鼻、目、耳、口
- **上半身**: 肩、肘、手首、指先、親指
- **下半身**: 腰、膝、足首、踵、つま先

### パラメータ調整

`MediaPipeDetector`クラスの主要なパラメータ：

```python
detector = MediaPipeDetector(
    static_image_mode=False,           # True: 静止画, False: 動画
    model_complexity=1,                # 0-2 (精度とスピードのトレードオフ)
    smooth_landmarks=True,             # ランドマーク平滑化
    min_detection_confidence=0.5,      # 検出の最小信頼度
    min_tracking_confidence=0.5,       # トラッキングの最小信頼度
)
```

### モデルの複雑さ (model_complexity)

- **0**: Lite - 最速だが精度は低い
- **1**: Full - バランス型（推奨）
- **2**: Heavy - 最高精度だが遅い

## 検証手順

### フェーズ1: 基本動作確認

1. Webカメラで骨格検出が正常に動作するか確認
2. 検出精度の目視確認
3. FPSの測定（リアルタイム処理可能か）

### フェーズ2: 精度評価

1. 様々な姿勢での検出テスト
   - 立位
   - 座位
   - しゃがみ
   - 手を上げる動作
2. 異なる照明条件でのテスト
   - 明るい環境
   - 暗い環境
   - 逆光

### フェーズ3: パフォーマンス測定

1. 処理速度（FPS）の測定
2. CPU使用率の確認
3. 異なる解像度での比較

### フェーズ4: 実用性評価

1. 作業者の動作追跡に十分な精度か
2. リアルタイム処理が可能か
3. 複数人の同時検出（今後の拡張性）

## トラブルシューティング

### MediaPipeのインポートエラー

```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

**原因**: Python 3.13を使用している、またはMediaPipeが正しくインストールされていない

**解決策**:
1. Python 3.12を使用していることを確認
   ```powershell
   python --version  # 3.12.x であることを確認
   ```

2. 仮想環境を再作成
   ```powershell
   # プロジェクトルートで実行
   Remove-Item -Recurse -Force .venv
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   cd skeleton-detection-validation
   uv pip install -e .
   ```

3. MediaPipeが正しくインストールされているか確認
   ```powershell
   python -c "import mediapipe as mp; print(mp.__version__)"
   # 0.10.9 と表示されればOK
   ```

### カメラが開けない

```
RuntimeError: カメラを開けませんでした
```

- カメラが他のアプリケーションで使用中でないか確認
- カメラのアクセス許可を確認
- `camera_id`を変更してみる（0, 1, 2など）

### 骨格が検出されない

- 照明が十分か確認
- 全身がカメラに収まっているか確認
- `min_detection_confidence`を下げてみる（例: 0.3）

### 処理が遅い

- `model_complexity`を下げる（1 → 0）
- カメラの解像度を下げる
- GPUが利用可能か確認

## 次のステップ

このプロジェクトでMediaPipeの有効性が確認できたら、以下を検討：

1. **作業動作の分類**: 特定の動作パターンの認識
2. **サイクルタイム測定**: 作業開始・終了の自動検出
3. **複数人対応**: 複数の作業者の同時追跡
4. **データベース連携**: 検出結果の保存と分析
5. **Web UI開発**: Streamlitでの可視化

## 参考資料

- [MediaPipe Pose公式ドキュメント](https://google.github.io/mediapipe/solutions/pose.html)
- [OpenCV公式ドキュメント](https://docs.opencv.org/)
- [OPEの受動取得](https://monoist.itmedia.co.jp/mn/articles/2110/27/news062_3.html)

## ライセンス

このプロジェクトは検証用です。

## 作成者

SeiyaTanaka@Classmethod (tanaka.seiya@classmethod.jp)
