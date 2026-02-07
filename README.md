# pinn_research

物理情報ニューラルネットワーク（PINN）の研究コードを管理するリポジトリです。
本プロジェクトでは、既存のPINNフレームワークに**重み付き損失関数（Weighted Loss）**の実装を追加し、その有効性を検証しています。

---

## 実行環境 (Execution Environment)

デバイスや用途に応じて、実行ファイルを選択してください。

| ファイル形式 | 実行環境 | 使用デバイス |
| :--- | :--- | :--- |
| `.ipynb` | Google Colab | TPU (Integrated) |
| `pinn.py` | ローカル環境 | GPU (NVIDIA GeForce RTX 3050) |

### セットアップ
Python環境で `pinn.py` を実行する場合、以下のコマンドで必要な依存ライブラリをインストールしてください。
```bash
pip install -r requirements.txt
