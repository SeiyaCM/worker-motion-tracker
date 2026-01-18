# worker-motion-tracker
カメラで人作業を撮影して自動的にサイクルタイムを取得する

## 技術スタック
| 項目 | 技術スタック | 備考 |
| :-- | :-- | :-- |
| 言語 | Python | - |
| Python Version | 3.12 | MediaPipe 0.10.9との互換性のため |
| backendフレームワーク | FastAPI | - |
| frontendフレームワーク | Streamlit | - |
| IaC | AWS CDK | - |
| プラットフォーム | Amazon Web Service | - |
| パッケージマネージャー | uv | - |
| CI/CD | CodeCommit, CodeBuild, CodeDeploy | - |
| 骨格検出 | MediaPipe + OpenCV | 作業者の動作追跡 |

## 参考資料
- [awslabs](https://github.com/awslabs/mcp/tree/main/src)
- [OPEの受動取得](https://monoist.itmedia.co.jp/mn/articles/2110/27/news062_3.html)