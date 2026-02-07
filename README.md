


# pinn_research

物理情報ニューラルネットワーク（PINN）の研究用リポジトリです。
本リポジトリでは、既存のPINNモデルをベースに、学習の最適化を目的とした**重み付き損失関数（Weighted Loss）**を独自に実装しています。

---

## 実行環境 (Execution Environment)

実行するファイルの種類によって、推奨されるハードウェア構成が異なります。

| ファイル形式 | 推奨環境 | デバイス | 備考 |
| :--- | :--- | :--- | :--- |
| `.ipynb` | Google Colab | **TPU** | Colab統合TPUを使用して動作確認済み |
| `pinn.py` | Local Python | **GPU** | NVIDIA GeForce RTX 3050 にて動作確認済み |

### インストール
Python環境で `pinn.py` を実行する前に、以下のコマンドで依存ライブラリをインストールしてください。
```bash
pip install -r requirements.txt

```

---

## 使用言語とパッケージ

### Python

メインの学習およびモデル実装に使用します。

### Julia

一部のスクリプトでJuliaを使用しています。可視化のために `Plots` パッケージが必要ですので、以下のコマンドで追加してください。

```julia
using Pkg
Pkg.add("Plots")

```

---

## 参照元 (References)

本プロジェクトのデータセットおよびコード構造は、以下のリポジトリを基にしています。

* **Dataset & Base Code:** [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs)
* **主な変更点:** 損失関数の計算において、各項の寄与を調整する **Weighted Loss（重み付き損失）** のロジックを追加実装しました。

---


