# worker-motion-tracker
カメラで人作業を撮影して自動的にサイクルタイムを取得する

## 技術スタック
| 項目 | 技術スタック | 備考 |
| :-- | :-- | :-- |
| 言語 | Python | - |
| Python Version | 3.11 | MediaPipe 0.10.9との互換性のため |
| backendフレームワーク | [FastAPI](https://fastapi.tiangolo.com/) | - |
| frontendフレームワーク | [Streamlit](https://streamlit.io/) | - |
| IaC | AWS CDK | - |
| テストフレームワーク | pytest | - |
| リンター/フォーマッター | Ruff | - |
| パッケージマネージャー | uv | - |
| プラットフォーム | Amazon Web Service | - |
| CI/CD | CodeCommit, CodeBuild, CodeDeploy | - |
| 認証、ユーザーディレクトリ | Amazon Cognito user pools | - |
| 骨格検出 | MediaPipe + OpenCV | 作業者の動作追跡 |

## 参考資料
- [awslabs](https://github.com/awslabs/mcp/tree/main/src)
- [OPEの受動取得](https://monoist.itmedia.co.jp/mn/articles/2110/27/news062_3.html)