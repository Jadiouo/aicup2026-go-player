# aicup2026-go-player

這是 AI CUP 2026 秋季賽「圍棋棋力與風格預測競賽」競賽一（Player）的程式碼。題目是給一批棋譜和一份候選人名單，找出下這些棋的人，交出前 5 名。

我們先把每盤棋變成能代表棋風的向量，再層層縮小候選人，最後讓答案盡量一對一。這個 repo 有資料處理、訓練、預測的程式，也有重現最終提交檔的 Colab 筆記本。

English: code for AI CUP 2026 Go, task 1 (identify the player behind a set of games). GPL-3.0.

另一題（預測棋力等級）放在 [aicup2026-go-rank](https://github.com/Jadiouo/aicup2026-go-rank)。兩個 repo 可以各自安裝、測試和執行。

## 成績

| | Public | Private |
|---|---|---|
| 名次 | 第 26 名 | 第 22 名 |
| 分數 | 0.8564817 | 0.8836589 |

每題交 5 個候選人，正解排第 r 名得 e^−(r−1) 分，榜上分數是各題平均。

## 方法概述

每題有 1 到 5 盤查詢棋譜和一份候選人池，要從池裡認出下這些棋的人。我們先用五種來源把每盤棋轉成特徵：手工特徵、KataGo HumanSL 落子統計、KataGo 主幹向量、逐局面棋風編碼器，以及從頭訓練的認人網路。接著用便宜的距離比對，從 600 位候選人中粗篩出 40 位，再用 LightGBM 排序模型在這 40 位之內細排。排序之後補上三個通道，分別是局面對局面比對、當場訓練的候選池分類器，和成對比對網路。最後用 CSLS 修正「誰都像」的候選人，讓答案盡量一對一。

提交檔共 1,600 列。新題 800 列走上面這條管線，舊題 800 列（候選庫每人 100 盤黑棋）沿用較早定案的另一條管線。

![Player 新題管線架構，橘框是用到外部預訓練模型的元件，其餘都是我們自己做的](docs/architecture.png)

## 快速開始

重現最終提交檔只需要 CPU，Colab 免費版就能跑完，約 5 分鐘。

1. 在 Colab 選 File → Open notebook → GitHub，貼上 `https://github.com/Jadiouo/aicup2026-go-player`，開啟 `colab/reproduce_final.ipynb`。
2. 把第一格的 `INTERMEDIATE_DIR` 指到放中間檔的資料夾，例如掛載的 Google Drive。`REPO_URL` 和 `RELEASE_URL` 已經填好。
3. Runtime → Run all。筆記本會安裝鎖定版本的套件，從 Release 下載自訓權重，驗證中間檔，然後跑最後一步。
4. 結尾會印出提交檔的 sha256，應該是 `b4ad40c24b5c6e991cc69bb08af060de23b3f68c96e64f5d7d79c36ca4c04643`。

自訓權重會自動從這個 repo 的 Release 下載。由官方棋譜算出來的中間檔不公開，需要向主辦單位索取，或照 [PIPELINE.md](PIPELINE.md) 自己產生。

舊題 800 列是 11/06 產生後保存下來的檔案，筆記本只核對它的 sha256，不重算。原因是缺 3 個已刪除的中間檔，細節在 PIPELINE.md 的「舊題 800 列」。

筆記本在本機用 `jupyter nbconvert --execute` 實際跑過，結果逐位元組相符。還沒有在真正的 Colab 上跑過。Colab 的 Python 和套件版本可能和鎖定版本不同，如果 pip 提示重新啟動，照做後從頭再跑一次。

本機安裝與跑測試：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## 目錄結構

```
README.md              本檔
PIPELINE.md            管線的步驟、輸入輸出、耗時，以及逐元件的訓練與推論指令
ASSETS.md              權重與中間檔的清單，含大小與 sha256
LICENSE                GPL-3.0 全文
requirements.txt       CPU 環境（鎖定版本）
requirements-gpu.txt   GPU 環境（torch 2.12.0 等）
docs/                  README 用的架構圖
src/gogame/            共用程式庫：SGF 解析、棋盤、特徵、聚合、模型、提交檔寫入與檢查
tests/                 單元測試，用合成棋譜，不含官方資料
scripts/               各階段腳本，eNN 是實驗編號，katago/ 是 HumanSL 掃描
final/                 最後一步：各通道分數組成 1,600 列提交檔
colab/                 Colab 筆記本、colab_utils.py、assets_manifest.json
```

`scripts/` 沒有再拆資料夾，因為腳本之間用相對位置 import，搬動會弄壞。

## 從頭重新訓練

從官方棋譜整條重跑，GPU 至少要約 76 小時。這是已知分項相加的結果，缺的分項沒有紀錄，實際只會更多，另外 HumanSL 掃描約 100 萬盤不算在內。Colab 免費版跑不完。

能直接重跑的是「中間檔到最終 CSV」這一段。更上游的每一步都有程式和指令，寫在 [PIPELINE.md](PIPELINE.md)。GPU 的結果不會位元重現，所以自己重算的中間檔和主辦方拿到的那份不會完全一樣，已知限制也列在 PIPELINE.md。

## 資料與權重

- 自訓權重共 17 個檔、128.4 MB，放在這個 repo 的 [Release v1.0](https://github.com/Jadiouo/aicup2026-go-player/releases/tag/v1.0)。
- 由官方棋譜算出的中間檔共 26 個檔、795.5 MB，不放在 repo 或 Release，可應主辦單位要求以私人連結提供。
- 官方棋譜本身不附。外部模型（KataGo、HumanSL、Strength Estimator）也不散布，下載網址與放置路徑見 [ASSETS.md](ASSETS.md) 第 3 節。
- 逐檔的大小與 sha256 在 ASSETS.md，機器可讀版是 `colab/assets_manifest.json`。

## 授權

本 repo 採 GPL-3.0，全文見 [LICENSE](LICENSE)。其中 `scripts/experiments/e21/se_encode.py` 依 Strength Estimator（GPL-3.0）和 MiniZero（Apache-2.0）公開原始碼的行為，用 Python 重寫了輸入平面轉換，並非逐字複製。選用 GPL-3.0 與此相容。

第三方元件不在本 repo 內，只有下載連結：

- KataGo 程式：MIT 式授權（Copyright David J Wu (lightvector) and/or other authors），`cpp/external` 內的第三方元件各有授權。
- KataGo 網路（kata1 與 Human SL）：KataGo Neural Network License，MIT 式。
- Strength Estimator：程式庫是 GPL-3.0，權重沒有看到明確授權。我們只從權利人的公開頁面下載使用，不再散布。
- MiniZero：Apache-2.0。
- 使用的開源套件：numpy、scipy、pandas、pyarrow、scikit-learn、LightGBM、PyTorch、sgfmill、PyYAML、numba。

## 生成式 AI 使用

開發過程使用 Claude Code（Anthropic）協助撰寫與審查程式、實驗與文件。
