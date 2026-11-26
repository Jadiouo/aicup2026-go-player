# Player（競賽一）管線

這份文件給想重現或讀懂整條管線的人。最終上傳的版本（內部代號 v7x）在 Private 拿到第 22 名、0.8836589。

提交檔 1,600 列，由兩半拼成，兩半用不同的方法。

- 舊題 800 列（`pub_`、`priv_`）：候選庫每人 100 盤黑棋。用的是代號 v6 的管線，內容是 E04v2 重排模型、SE 與 HumanSL 的距離融合，再加轉導解碼。這半邊是 11/06 產生後保存下來的檔案 `old_v6_800.csv`，執行順序、輸入輸出和缺件見下方「舊題 800 列」。
- 新題 800 列（`pub2_`、`priv2_`）：候選庫 600 人、每人 10 盤黑白混，查詢 1、2、3、5 盤。這半邊是從 v7a 一路疊加到 v7x 的管線，也是本文件的重點。

指令預設以 repo 根目錄為工作目錄，並設定 `PYTHONPATH=src:scripts:.`。腳本頂端的 docstring 有完整參數。

## 代號對照

程式與紀錄用實驗編號命名。腳本放在 `scripts/experiments/eNN/` 和 `scripts/oneoff/eNN_*.py`。下表是這份文件會用到的編號。

| 代號 | 是什麼 |
|---|---|
| `e13` | KataGo 主幹向量的編碼器 |
| `e21` | Strength Estimator（SE，中研院的棋力估計模型）特徵抽取 |
| `e27` | 舊題 800 列的融合管線 |
| `e29` | 訓練資料包（逐盤特徵與折角色） |
| `e34`、`e35`、`e36` | HumanSL 後半盤（第 61 到 120 手）掃描與 W120 表 |
| `e38`、`e39`、`e40`、`e41` | 白化 LDA 度量與深度度量編碼器 |
| `e52`、`e56`、`e62`、`e69` | 四代主幹向量抽取 |
| `e77`、`e80`、`e86` | 逐局面棋風編碼器（pe86）的三代 |
| `e89`、`e90` | S 認人網路的資料、訓練與 booster |
| `e99` | 大型認人網路（16 區塊×224 通道） |
| `e110` | CSLS 一對一先驗的驗證 |
| `e112` | 局面對局面比對通道（輸出 X_e112） |
| `e123` | 候選池分類器通道（輸出 X_lr） |
| `e127` | 成對比對網路通道（輸出 X_p1r） |

其他名稱：

- W60、W120 是 HumanSL 掃描的視窗，分別取前 60 手和前 120 手。
- pe86 是逐局面棋風編碼器，把一盤棋每個局面的向量聚合成 128 維。
- `finh_full`、`finh_h0`、`finh_h1` 是三個 S 認人網路，後兩個各用一半玩家訓練，供交叉擬合。
- v7q 是 booster，也就是 LightGBM 排序模型，輸入 792 個特徵。
- v3 是舊題的基底版本，v6 是它加上 SE 與 HumanSL 融合之後的版本。

## 環境與路徑

CPU 環境用來做特徵、LightGBM、sklearn 和最終組合，Python 3.12。GPU 環境只有 torch 的步驟需要。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# GPU 環境：系統 python3 加 torch 2.12.0+cu130，在 Colab 可以直接用內建的 torch
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pytest -q
```

所有腳本以 repo 根目錄為工作目錄，用 `PYTHONPATH=src:scripts:.` 執行，例如：

```bash
PYTHONPATH=src:scripts:. .venv/bin/python scripts/predict.py ...
```

scikit-learn 必須是 1.9.1，Player 的候選池分類頭只有在 sklearn 1.9.1 加預設 tol 下才位元重現。LightGBM 用 4.7.0。

路徑約定：

- `data/` 放官方資料與掃描結果，不附。環境變數 `DATA_DIR` 可以指到別處，最終組合腳本支援。
- `.scratch/` 是工作目錄，放所有中間檔與權重。環境變數 `WORK_DIR` 可以指到別處，最終組合腳本支援。
- 較上游的腳本直接用 repo 根目錄下的 `data/` 和 `.scratch/`。放在別的磁碟時，在 repo 根目錄建同名符號連結，例如 `ln -s /big/disk/data data`。
- `e79_items.py` 和 `e79_prep_local.py` 的 ssh／rsync 目標由環境變數 `REMOTE_HOST` 指定。Colab 控制腳本的 CLI 由 `COLAB_CLI` 指定。
- `tests/fixtures/golden/` 的棋譜是 `tests/tools/make_synthetic_games.py` 產生的隨機合成棋譜，不是官方資料。部分測試（約 13 個）需要官方資料，沒有資料時會被 skip。

官方資料由主辦方提供，不附。請放在 `data/` 下：

```
data/training/       train_<rank>.csv …，訓練棋譜 1,000,000 盤（欄位 player_id, game_id, rank, color, sgf_content）
data/contest_1104/   原測試集：player_identification_test_{public,private}.csv、player_identification_candidates_{public,private}.csv（舊題）
data/contest_1111/   11/11 新增的 Player 測試集：player_identification_test2_*.csv、player_identification_candidates2_*.csv
```

## 管線總覽與步驟表

下圖由上往下讀。上面是各種特徵，下面是逐層縮小候選人的過程。

```
官方棋譜 ─┬─ L0 手工特徵 ────────────────────────────────┐
          ├─ KataGo HumanSL 掃描（W60、W120）────────────┤
          ├─ KataGo 主幹向量（逐盤、逐局面）→ 逐局面編碼器 pe86 ─┤→ 第一階段：多通道融合取前 K=40 位候選人
          └─ S 認人網路（finh_full / finh_h0 / finh_h1）──┘        │
                                                                    ▼
          第二階段：LightGBM lambdarank booster（792 特徵）在短名單內重排 → e
                                                                    ▼
          c = rowz(e) + 0.5·X_e112 + 1.0·X_lr + 0.35·X_p1r   （三個外掛通道）
          c' = CSLS(c; k=3, α=.5)（同池 400 題轉導，軟性一對一先驗）→ 取前 5 → 與舊題 800 列拼接
```

步驟表依序列出每一步的腳本、輸入輸出和硬體。

| # | 階段 | 腳本 | 輸入 → 輸出 | 硬體／耗時 |
|---|---|---|---|---|
| 1 | 資料處理：SGF、L0 | `src/gogame/{sgf,board,features_l0}.py`、`scripts/build_l0_cache.py`、`scripts/make_split.py`、`scripts/predict.py` | 棋譜 → L0 稠密／稀疏特徵；以 player_id 分組的切分 | CPU，分鐘到小時 |
| 2 | 特徵：HumanSL 掃描 | `scripts/katago/{scan_setup.sh,setup_katago.sh,scan_runner.py,scan_merge.py,scan_pins.json}`；測試端 `scripts/oneoff/p1111_hsl_prepare.py`、`scripts/experiments/e34/scan_runner_w2.py`（後半盤 W120） | SGF → 逐手 log p(落子)、名次、熵（11 個段位 profile，1 visit）→ `data/test_scan_p1111/hsl*_test_games.parquet` | GPU（KataGo CUDA）。訓練端約 100 萬盤：Colab L4 第 1 層 43.8 units（約 2.35 盤／秒），加學校 RTX8000 第 2 層約 80 萬盤 |
| 3 | 特徵：KataGo 主幹向量 | `scripts/experiments/e13/{encode,train}.py`（`KataEncoder`、`load_katago_trunk`）、`scripts/oneoff/{e52_kata_trunk2,e56_trunk,e62_trunk,e69_trunk}.py` | SGF → 逐盤向量（落子點 384 維，加全盤平均 384 維） | GPU，本機約 1,200 局面／秒 |
| 4 | 編碼：逐局面編碼器 pe86 | `scripts/oneoff/e77_enc.py`、`e80_{enc,extract,final}.py`、`e86_{prep,gpu,colab_prep,colab_enc,final}.py`、`e79_{items,prep_local,tok,gpu}.py` | 全部己方局面的主幹向量（PCA-64）→ 每盤 128 維 | GPU；全局面抽取 Colab L4 9.2 小時，編碼器每個 12.6 分鐘 |
| 5 | 訓練：S 認人網路 | `scripts/experiments/e89/{data_prep,train_s,fastboard,driver}.py`、`scripts/experiments/e90/{make_data_h,train_e90h,driver_e90h,assemble_pe_h}.py` | 約 100 萬盤（18 個 19×19 平面，10 區塊×128 通道 ResNet，cosine 分類 7,365 類）→ `finh_{full,h0,h1}/final_model.pt`、每盤向量 `pe_test_{pub,priv}.npy` | GPU；`finh_full` Colab L4 20.8 小時、`finh_h1` 14.4 小時、`finh_h0` 本機 5070 Ti 6.2 小時 |
| 6 | 訓練＋推論：度量、深度編碼器、booster | `scripts/oneoff/{e38,e39,e40,e41,e52,e62}_final.py`、`scripts/experiments/e90/{e90_final,e90_final_core,e90_feats2}.py` | 白化 LDA 度量、InfoNCE 深度度量、cross-fit。60 個模擬池×400 題×40 候選＝960,000 列訓練 booster（lambdarank，600 輪，seed 8）→ `real_v7q_S_{pub,priv}.npz`（e、top、pool、qids）和 `models/v7q_S.*` | CPU（booster 約 13 分鐘；度量與編碼器在 GPU 各約 14 分鐘） |
| 7 | 推論：X_e112（局面核密度比對） | `scripts/experiments/e112/e112_{export,lib,stage2,real,amend1}.py` | 三個 S 網路的逐局面向量 → `real_X_{pub,priv}.npz`（400×600） | GPU 匯出，加 CPU |
| 8 | 推論：X_lr（候選池分類頭） | `scripts/experiments/e123/e123_{lib,make,val}.py` | 每池 6,000 盤向量 → 600 類 sklearn LogisticRegression（C=1，lbfgs）→ `real_Xlr_{pub,priv}.npz` | CPU（需 sklearn 1.9.1） |
| 9 | 訓練＋推論：成對比對網路 X_p1r | `scripts/experiments/e127/{make_rows,e127_export,e127_data,e127_model,e127_train,e127_x,e127_eval0,e127_score}.py`、`real_x.py`、`job_export.sh`、`job_train.sh`、`job_x.sh`、`job_real_x.sh` | 逐局面向量匯出 → 訓練（12,000 步、lr 1e-3、seed 0）→ 對真實 test2 推論 → `real_Xp1r_{pub,priv}.npz` | GPU（本機 5070 Ti，0.04 s／步、VRAM 7.2 GB，每網路約 8 分鐘） |
| 10 | 最終組合 | `final/make_player_final.py`（原型：`scripts/experiments/e127/make_final.py`、`scripts/experiments/e123/e123_make.py`） | 上述 npz 加 `old_v6_800.csv` → 1,600 列提交檔；並用 `scripts/oneoff/p1111_assemble.py` 驗證 | CPU，數秒 |

步驟 9 的訓練指令（原紀錄）：

```bash
bash scripts/experiments/e127/job_train.sh p1r cross "--steps 12000 --lr 1e-3 --lr_p 3e-4 --B 8 --M 24,58 --mon_every 1000 --ckpt_every 2000 --cand_sa 1 --qry_sa 1 --drop 0.1 --resid 1"
```

`job_*.sh` 用系統 `python3` 和 torch。原作者用本機 GPU 排隊工具 `gpujob` 和 `systemd-run` 的記憶體上限來跑，這兩者不是必要的。

逐元件的訓練與推論指令、輸入、輸出，在本文件最後的「元件手冊」。權重與中間檔的大小、sha256 和放置位置見 [ASSETS.md](ASSETS.md)。Colab 版的最後一步見 `colab/reproduce_final.ipynb`。

## 最後一步

這一步把各通道的分數組成 1,600 列提交檔，只用 CPU，數秒就跑完。

```bash
.venv/bin/python final/make_player_final.py \
    --work-dir <WORK_DIR> --out out/player_v7x.csv [--w-new 0.35] [--reference 上傳檔.csv]
# 環境變數 DATA_DIR 指向含 contest_1111/ 的資料夾（驗證用）
```

`--reference` 會逐位元組比對上傳檔並印出 sha256。`--w-new 0` 得到前一版 v7w（sha256 前綴 `21a6fde0`），可以交叉驗證舊題加三通道的部分。沒有中間檔的人，可以讀程式了解最後一步的公式，再依本文件往上游重算。

最後一步需要的輸入，都在 `<WORK_DIR>` 下：

| 輸入 | 內容 |
|---|---|
| `e90/final/make/real_v7q_S_{pub,priv}.npz` | `e`（400×40 booster 分數）、`top`（短名單）、`pool`（候選人 id）、`qids` |
| `e112/final/real_X_{pub,priv}.npz` | `X` 400×600 |
| `e123/final/real_Xlr_{pub,priv}.npz` | `X` 400×600 |
| `e127/final/real_Xp1r_{pub,priv}.npz` | `X` 400×600、`ng` |
| `e62/old_v6_800.csv` | 舊題 800 列（`--old800` 可改） |

輸出是 `--out` 指定的 1,600 列 CSV，欄位為 `question_id,top_1..top_5`。

## 舊題 800 列

提交檔前一半（`pub_`、`priv_`，候選庫每人 100 盤黑棋，查詢 5、10、15、20 盤各 100 題）是 v6。它在 2026-11-06 產生一次（sha256 `b4eed172f033c7a20431c439216a4678022342765615d79610a1851764e98d10`），之後 v7j2 起的每個版本都原樣沿用。

目前保存檔可以核對（`final/check_old_v6.py`），但由程式重算最後一步跑不了，因為缺 3 個已被刪除的中間檔，見下方執行順序表。

方法分三步：

1. 基底 v3 是 E04v2 LightGBM lambdarank 重排（L0 特徵前 20 名，586 欄，程式在 `src/gogame/player_rerank.py`），再加轉導解碼。
2. 融合 e′ = e − 0.5·λ_se·D_se − 0.5·λ_hsl·D_hsl，其中 λ_se = 0.5、λ_hsl = 0.25（實際係數 0.25、0.125）。D 是白化 LDA 空間的馬氏距離平方，再除以 (1/N_q + 1/n_lib)，度量只用 fit 玩家擬合。SE 用 `se_mean` 一維。HSL 對每位候選人隨機取 20 盤（`default_rng(20)`），查詢盤全取。
3. 轉導解碼：互為最佳且 softmax ≥ τ = 0.9 的配對先定案、移出後重算，做 3 輪，再做 Hungarian 一對一（`scripts/oneoff/player_sim3.py: trans_t5`）。

執行順序如下。指令要設 `GOGAME_COLOR_SEMANTICS=actual`，並以 repo 根目錄為工作目錄、`PYTHONPATH=src:scripts:.`。

| # | 指令 | 輸入 → 輸出 | 硬體／耗時 | 現況 |
|---|---|---|---|---|
| 1 | `python scripts/predict.py train --task player --player-model e04v2 --include-holdout`（需先做步驟 1 的切分與 L0 快取） | 訓練棋譜 → `data/models_final/player_e04v2.pkl`（含 H 重訓版） | CPU | 權重在 Release（ASSETS.md「自訓權重」） |
| 2 | `python scripts/oneoff/e27_player_fusion.py v3`（可設 `V3_REF=<v3 提交檔>` 逐列比對） | `data/contest_1104/*` 加步驟 1 的模型 → `.scratch/e27/v3_repro.csv` | CPU，約 1.5 分鐘 | 已在公開 repo 的副本實跑，sha256 前 12 碼 `b51ae1671927`，與 v3 上傳檔逐列相同（2026-11-26） |
| 3 | `python scripts/oneoff/player_sim3.py prep` | 訓練端 SE 表 `.scratch/e21/se_uid.parquet`（`scripts/experiments/e21/extract.py`，GPU 12.9 小時）、`data/cache/e10/hsl_games.parquet`、`data/cache/games_meta.parquet`、切分 → `.scratch/player_sim3/metric_se.npz`、`metric_hsl.npz` | CPU（SE 表要 GPU） | 缺：兩個度量檔與 `se_uid.parquet` 已刪（HSL 的來源表在，SE 的來源表不在） |
| 4 | `python scripts/oneoff/e27_player_fusion.py games` | `data/contest_1104/*` → `.scratch/e27/se_games.parquet`（90,000 盤）、`hsl_games_in.parquet`（26,000 盤） | CPU | 可重跑 |
| 5 | 把 `se_games.parquet` 切成 9 個 10,000 盤分塊 `.scratch/e27/in/c{0..8}.parquet`，逐塊 `python scripts/predict_se.py extract --games-parquet … --out-dir .scratch/e27/se_out/c{k} --model <se_go.pt> --device cuda --half`，再 `python scripts/oneoff/e27_merge_se.py` | → `.scratch/e27/se/se_test_games.parquet`（`uid,se_mean,se_std,se_n`） | GPU，每塊約 7 分鐘（RTX8000），合計約 1 小時 | 缺：此表已刪，可重抽（`colab/full_pipeline.ipynb` 段 7 有程式） |
| 6 | `python scripts/oneoff/e27_hsl_prepare.py data/test_scan_e27player`，然後 `scripts/katago/scan_runner.py`（分片可續跑）→ 入帳 → `python scripts/predict_hsl.py --root data/test_scan_e27player features` → 複製 `hsl_test_games.parquet`、`features_manifest.json` 到 `.scratch/e27/hsl/` | 26,000 盤 → HSL 特徵表（104 欄加 uid） | GPU，約 4 小時（RTX8000，1.9 盤／秒） | 特徵表是官方資料衍生的中間檔，僅提供主辦方（ASSETS.md 的 `e27player_hsl_test_games.parquet`，14.4 MB） |
| 7 | `python scripts/oneoff/e27_player_fusion.py fuse [<含 se/、hsl/ 的資料夾> <輸出 CSV>]` | 步驟 3、5、6 的輸出加步驟 1 的模型 → 800 列 CSV（即 `old_v6_800.csv`）和 `fuse_summary.json` | CPU，約 1.5 分鐘 | 目前無法執行（缺步驟 3、5 的檔） |

保存檔與紀錄是否一致，可以在沒有上游檔的情況下核對：

```bash
python final/check_old_v6.py <old_v6_800.csv> [--v3 <v3 提交檔>]
# 預期：sha256 前 12 碼 b4eed172f033；800 列格式；與 v3 相比 top-1 變 10 題、任一名次變 795 題
```

top-1 改變的 10 題，是融合當時紀錄的數字，依 N = 5、10、15、20 分別為 6、2、1、1 題。2026-11-26 實測 sha256 相符、格式 0 錯誤，與 v3 的差異數也和紀錄相同（10／795）。

其他注意事項：

- 基底模型用含 H 玩家重訓的 `player_e04v2.pkl`。λ 是用不含 H 的模擬池選的（`tune_combo_table.csv` 中 tune 分數最高者），測試端沒有再調。
- 離線模擬 dev 40 池從 .9927 到 .9967（Δ +.0040），模擬與真實 Public 差約 .007。舊題這一半的 Public 約 1.000，這是推論值，不是實測。
- 新題那一半不用這個轉導解碼。新題結構未知，轉導與 Hungarian 在「同人多題」的結構下大幅變差。
- SE 輸入編碼 `scripts/experiments/e21/se_encode.py` 與 SE 權重的授權說明，見 README 的「授權」。

## 沒有中間檔時怎麼重新產生

最後一步和多數上游步驟的輸入，是由官方棋譜算出來的：各通道對測試題的分數矩陣、逐局面與逐盤向量、舊題 800 列保存檔。這些檔案不放在 repo 或 Release，只能向主辦單位索取。

沒有這些檔時，可以從官方資料、我們的權重和外部模型重新產生。下表的硬體和時間是原作者實測。GPU 的結果不會位元重現，所以重算出來的檔和主辦方那份不會位元組相同，但最後一步的算式一樣。

| 中間檔 | 重新產生需要 | 腳本（參數見檔頭說明） | 硬體與時間 | 備註 |
|---|---|---|---|---|
| `real_v7q_S_{pub,priv}.npz`（booster 分數與短名單） | 官方測試檔；L0、HSL、主幹向量、S 向量、pe 向量；Release 的 `v7q_S.*` | `scripts/experiments/e90/e90_final.py`（produce；輸入由 `p1111_*`、`e86_final.py` 等產生） | 抽特徵用 GPU；booster 推論與訓練用 CPU，約 13 分鐘 | 依賴 pe86 向量。pe86 編碼器權重沒有保存，需重訓（每個 12.6 分鐘，加訓練端全局面抽取約 9.2 小時） |
| `real_X_{pub,priv}.npz`（X_e112） | 官方測試檔；Release 的 `finh_full`、`h0`、`h1` | `scripts/experiments/e112/e112_export.py` → `e112_real.py` | GPU 匯出逐局面向量，加 CPU | 只做推論，不重訓 |
| `real_Xlr_{pub,priv}.npz`（X_lr） | 每盤向量（小 S、大 S、pe86） | `scripts/experiments/e123/e123_make.py` | CPU（需 scikit-learn 1.9.1） | 每池 600 類 LogisticRegression |
| `real_Xp1r_{pub,priv}.npz`（X_p1r） | 測試端逐局面向量；Release 的 `p1r_*`、`sigma_*` | `scripts/experiments/e127/e127_export.py`、`real_x.py`、`job_real_x.sh` | GPU（本機 5070 Ti，VRAM 7.2 GB）；若重訓，每網路約 8 分鐘 | 推論用 Release 權重 |
| `pos_*`、`t2_pos_*`、`pe_test_*`、`Z_*`（逐局面與逐盤向量、L0 特徵） | 官方測試棋譜；KataGo 主幹（外部）；Release 的 S 網路 | `scripts/oneoff/e86_*`、`e80_*`、`scripts/experiments/e90/embed.py`、`e90/assemble_pe_h.py`、`scripts/oneoff/p1111_final.py` | GPU；主幹向量本機約 1,200 局面／秒 | 見上方步驟 3 到 5 |
| `e27player_hsl_test_games.parquet`、`old_v6_800.csv`、v3 與 v7 參考 CSV | 舊題 v6 | `scripts/oneoff/e27_*.py`、`player_sim3.py` | HSL 約 4 小時、SE 約 1 小時（RTX8000） | v6 最後一步目前無法重跑（缺 3 個已刪除的檔，見「舊題 800 列」） |

## 外部模型的用途

本專案沒有使用任何外部棋譜或標註資料，外部資源只有下列預訓練模型和開源套件。這些檔案不在 repo，也不在 Release，請自行從官方網址下載並遵守授權，下載後用 sha256 核對。網址、大小、sha256 和放置路徑在 [ASSETS.md](ASSETS.md) 第 3 節。各網址於 2026-11-26 確認可連線，檔案大小與表中相同。

在 Colab 上，`colab/colab_utils.py` 的 `fetch_external()` 可以一次下載並驗證，也就是 `colab/reproduce_final.ipynb` 的第 3 格，預設略過。KataGo 另可用 `scripts/katago/setup_katago.sh`（Colab）或 `scan_setup.sh`。

各資源怎麼用：

- KataGo 執行檔：以 1 visit 對每盤棋逐手分析，取落子機率（HumanSL 掃描）。不微調，也不當訓練資料。
- KataGo 主網路：KataGo 載入 HumanSL 時必須一併載入。
- KataGo HumanSL 網路：提供 11 個段位 profile 的逐手人類落子機率，是核心特徵。授權和上面一致，但訓練棋譜的來源作者沒有公開說明。
- HumanSL PyTorch 檢查點：抽主幹向量（固定 profile `rank_1d`、日本規則）。載入前要用 KataGo 原始碼的 `migrate_defaultdict_to_dict.py` 轉換（torch 2.12）。
- Strength Estimator 權重（中研院）：每盤取 32 個己方局面，取 `score` 頭的平均與標準差，用 `torch.jit.load` 載入（`scripts/predict_se.py`、`scripts/experiments/e21/extract.py`）。只當黑盒特徵使用。權重沒有明確授權，我們只從公開頁面下載使用。

## 驗證設計

一律以 player_id 分組切分，同一位玩家不會同時出現在訓練與評估。新題沒有答案，離線改用「真實候選庫偽查詢」評量：每人 10 盤中 6 盤當庫、4 盤當查詢，再加 16 個模擬池。

## 元件手冊

上面的步驟表是由下到上的流程，這一節則是逐元件的指令。

- 指令以 repo 根目錄為工作目錄。CPU 腳本用 `PYTHONPATH=src:scripts:. .venv/bin/python`。標「GPU」的用系統 `python3` 加 torch（`requirements-gpu.txt`）。
- 完整參數看各腳本頂端的 docstring。`.scratch/` 和 `data/` 的位置見上面「環境與路徑」。
- 「報告名稱」是報告正文用的描述性名稱，「內部代號」是程式與紀錄裡的編號。
- 原作者用本機 GPU 排隊工具和 `systemd-run` 的記憶體上限包住重工作，這兩者不是必要的。

### 資料與特徵（多半無訓練）

#### 1. 手工特徵（L0）

無訓練。

- 抽取：`.venv/bin/python scripts/build_l0_cache.py [--jobs 14] [--chunk 20000]`。測試端由 `scripts/oneoff/e38_final.py predict` 即時呼叫 `src/gogame/features_l0.py`，結果存 `.scratch/p1111/z_t2/Z_{pub,priv}.npz`。
- 輸入：官方棋譜（`data/training*`、`data/contest_1111/`）。輸出：`data/cache/l0/{dense.parquet,sparse.npz}`。
- 官方測試題的欄位與 id 整理：`.venv/bin/python scripts/oneoff/p1111_inspect.py --data-dir data/contest_1111`，寫到 `.scratch/p1111/normalized/`。

#### 2. 訓練資料包（E29 bundle）

無訓練。

```bash
.venv/bin/python -m scripts.experiments.e29.prep_bundle
```

輸出 `.scratch/e29/bundle/{X.npy,aux.npz,manifest.json}`，內容是非保留集玩家的逐盤特徵與折角色。後面的 `e36`、`e40_finalprep`、`e55_prep` 都讀它。

#### 3. HumanSL 落子統計（前 60 手、前 120 手）

用外部的 KataGo HumanSL，無訓練。

- 抽取（訓練端）：`bash scripts/katago/scan_setup.sh` 會下載並以 sha256 釘選 KataGo 與網路。接著跑 `scripts/katago/scan_runner.py`（前 60 手）和 `scripts/experiments/e34/scan_runner_w2.py`（第 61 到 120 手），參數見兩者的 docstring。合成訓練端 W120 表：`python -m scripts.experiments.e36.build_w120_full old|new|train`，輸出 `.scratch/e36/bundle/X120_full.npy`。
- 抽取（測試端）：先 `GOGAME_COLOR_SEMANTICS=actual .venv/bin/python scripts/oneoff/p1111_hsl_prepare.py data/test_scan_p1111`，再 `scripts/predict_hsl.py --root data/test_scan_p1111 run --backend school ...`，然後 `scripts/predict_hsl.py --root ... features`。後半盤合併用 `.venv/bin/python scripts/oneoff/e41_late_merge.py`，輸出 `data/test_scan_p1111/hsl120_test_games.parquet`（uid 加 104 欄）。
- 輸入：SGF。輸出：逐手機率、名次、熵，再彙總成逐盤 104 欄。

#### 4. KataGo 主幹向量（逐盤）

這是外部預訓練網路的內部啟動值，無訓練。共四代，編號 E52、E56、E62、E69。每代都是先在 CPU 準備局面，再用 GPU 抽向量。

- 前半盤、6 局面（E52）：`.venv/bin/python scripts/oneoff/e52_kata_prep3.py`，然後 GPU `E52_KDIR=kata3 python3 scripts/oneoff/e52_kata_trunk2.py`。
- 前半盤、12 局面、多層（E56）：`.venv/bin/python scripts/oneoff/e56_prep.py --set final`，然後 GPU `python3 scripts/oneoff/e56_trunk.py --set final`。
- 後半盤（E62）：`e62_prep.py --set final`，然後 GPU `e62_trunk.py --set final`。
- 多區塊 3、6、12、15（E69）：`e69_prep.py --set final`，然後 GPU `e69_trunk.py --set final`。
- 輸入：SGF 和 KataGo 的 `b18c384nbt-humanv0.ckpt`（外部，見 ASSETS.md）。
- 輸出：`.scratch/e52/kata3/`、`.scratch/e56/final/`、`.scratch/e62/final/`、`.scratch/e69/final/` 下的 `blk{i}_gv.npy`（逐盤向量）。編碼器在 `scripts/experiments/e13/encode.py`。

### 神經網路

#### 5. 逐局面棋風編碼器（pe86，內部代號 E77、E80、E86）

訓練 6 個編碼器：全部、甲半、乙半，各有「模擬池」與「最終」兩組。最終只用其中 3 個。

- 準備：`python scripts/oneoff/e77_prep_full.py --set pca` 擬合 PCA 局面，再 GPU `python3 scripts/oneoff/e77_gpu_full.py --sets pca`，寫出 `.scratch/e77/pca_full.npz`（PCA 96 維，E80 與 E86 取前 64 維）。
- 前後視窗的 24 個 token（E79）：`e79_items.py`、`e79_prep_local.py`、GPU `e79_gpu.py`、`e79_tok.py`。另有 `e80_meta.py`、`e80_prep.py` 和 GPU `e80_extract.py`。
- 全局面（E86）：`e86_items.py`、`e86_meta.py`，再 `e86_prep.py STAGE ITEMS.pkl` 加 GPU `e86_gpu.py STAGE`，或用 Colab 版 `e86_colab_prep.py`。
- 訓練並匯出（最終）：`python scripts/oneoff/e86_colab_enc.py [--only sim,final]`，一次訓練六個編碼器並匯出。較早的 E80 版是 `python3 scripts/oneoff/e80_enc.py sim|final [--only h0,h1,full]`（24 個 token）。編碼器定義與訓練函式在 `scripts/oneoff/e77_enc.py`。
- 推論：同一批腳本會匯出 `pe_final_h0.npy`、`pe_final_h1.npy`（訓練池用）和 `pe_test_{pub,priv}.npy`（測試用）。
- 輸入：全部己方局面的主幹向量（PCA-64）。輸出：每盤 128 維單位向量。權重是 `enc_*.pt`，已經不在了，只剩向量（見 ASSETS.md）。

#### 6. 認人網路（S，內部代號 E89、E90）

共三個：`finh_full`、`finh_h0`、`finh_h1`，都是 10 區塊×128 通道的 ResNet。

- 資料：先 `PYTHONPATH=src:. .venv/bin/python scripts/experiments/e90/make_data.py union|test|meta`，得到 `u_games.npz`、`rosters.npz`、`t_games.npz`、`export_spec*`。再 `.venv/bin/python scripts/experiments/e90/make_data_h.py build`，最後 `python scripts/experiments/e90/make_data_h.py concat [--dir DIR]` 加入保留集的盤，得到 `uh_games.npz`、`rosters_h.npz`、`export_spec_h.*`。自寫的棋盤引擎在 `scripts/experiments/e89/fastboard.py`（numba）。
- 訓練（GPU，可續跑）：`python scripts/experiments/e90/driver_e90h.py --roster finh_full --root DIR --accum 2 --workers 8 --export auto`。`finh_h0`、`finh_h1` 同理。DIR 內要放 `uh_games.npz rosters_h.npz t_games.npz export_spec_h.json export_spec_h.npz`。驅動器會呼叫 `train_e90h.py`（20 輪、種子 0）。
- 推論（逐盤向量）：訓練結束時 `--export auto` 會一併寫出 `vec_*.npz`。整理成 booster 用的檔：`python scripts/experiments/e90/assemble_pe_h.py DIR .scratch/e90/r_h`，得到 `pe_final_h0.npy`、`pe_final_h1.npy`、`pe_test_{pub,priv}.npy`。逐局面向量匯出見元件 13。
- 輸入：約 100 萬盤的己方局面，共 18 個 19×19 平面。輸出：`finh_*/final_model.pt`（權重）和每盤 128 維向量。

#### 7. 大型認人網路（E99，16 區塊×224 通道，共三個）

只當候選池分類器的向量來源，不進重排模型。

- 訓練（GPU，可續跑，資料與元件 6 相同）：`python scripts/experiments/e99/driver_e99.py --roster finh_full --root DIR --epochs 12`。`finh_h0`、`finh_h1` 同理。DIR 內的檔案同元件 6。每個網路用 `--accum 8`，種子 0 由驅動器固定。訓練本體是 `scripts/experiments/e99/train_e99.py`，它是 `train_e90h.py` 的複本，只改了規模。
- 推論（逐盤向量）：訓練時 `--export auto` 會寫 `vec_*.npz`。再跑 `python scripts/experiments/e99/assemble_pe_e99.py [NETS_DIR] [OUT_DIR]`，預設是 `.scratch/e99/nets` 到 `.scratch/e99/r`。
- 輸入：同元件 6。輸出：`.scratch/e99/nets/finh_full/final_model.pt`（Release 檔名 `finh_bigS_full__final_model.pt`）和 `.scratch/e99/r/pe_test_{pub,priv}.npy`。
- 另外，`e99_final.py` 是「用大網路取代或加入小網路」的重排模型（v7r）的產檔程式，`test_feats3.py` 是它的特徵測試。v7r 兩個版本都沒贏，沒有上傳，最終管線不用。

#### 8. 深度度量編碼器（E40、E41）

兩個 MLP 編碼器，各有全部、甲半、乙半三版，用 episodic 對比學習訓練。

- 準備：`.venv/bin/python scripts/oneoff/e40_finalprep.py` 產生最終 Z 空間與標準化後的 HumanSL。Z 被清掉的話用 `e41_finz.py` 重建。接著跑 `e40_fin_inputs.py` 和 `e41_fin_inputs120.py`（測試端輸入）。
- 訓練並嵌入（GPU）：`python3 scripts/oneoff/e40_deep_mlp.py --data fin --hsl 1 --out fin_full --half -1`（前 60 手 HumanSL 版），以及 `python3 scripts/oneoff/e41_deep_fin.py --data fin --hsl 2 --out fin120_full --half -1`（前 120 手版）。`--half 0|1` 訓練甲半或乙半版。設定是 30,000 步、hidden 4096、P=512、dim 256，詳見 E40、E41 的紀錄。
- 集成：兩個編碼器串接後除以 √2，得到集成嵌入：`python scripts/oneoff/e50_ens.py --w60-dir .scratch/e40 --w60-tag fin --out finens --out-dir .scratch/e41`。
- 輸入：L0 的 Z 和標準化後的 HumanSL。輸出：`enc_*.pt`、`emb_finens_{h0,h1}_cap.npy`、`emb_finens_full_test_{pub,priv}.npy`。
- 甲乙半分組檔 `.scratch/e40/fin/halves.npz` 在原專案是當時用一次性指令建立的，沒有獨立腳本。

### 統計模型與重排

#### 9. 白化線性判別度量（LDA 度量）與召回融合

無神經網路，在重排模型訓練時一併擬合（同一支腳本，見元件 10）。包含 L0 LDA（150 維、ridge 3e-2）、HumanSL LDA（15 維）和主幹向量 LDA。用馬氏距離的 z 分數相加，取前 K=40 位候選人。程式在 `scripts/oneoff/e38_final.py` 和 `e39_final.py`（交叉擬合分組）。

#### 10. 重排模型（LightGBM lambdarank，v7q，792 特徵）

訓練與推論在同一支腳本。

- 訓練並做測試端預測：`PYTHONPATH=src:scripts:. .venv/bin/python scripts/experiments/e90/e90_final.py make --s-dir .scratch/e90/r_h`。它經 `e90_final_core.py` 使用 `scripts/oneoff/e86_final.py` 的訓練與預測碼，用 60 個模擬池×400 題×40 候選人（96 萬列）、600 輪、種子 8，Public 與 Private 池各自預測。`e90_final.py preflight --s-dir ...` 可以先做不吃記憶體的檢查。
- 輸入：元件 1、3、4、5、6、8 的輸出，加 `.scratch/e90/r_h`（認人網路向量）和 `.scratch/e86/r`（逐局面編碼器向量）。
- 輸出：`.scratch/e90/final/make/real_v7q_S_{pub,priv}.npz`（`e`、`top`、`pool`、`qids`）、`models/v7q_S.{txt,json}` 和 5 個度量 npz。

#### 11. 局面對局面比對（E112，核密度，無訓練）

- 匯出逐局面向量（GPU）：`bash scripts/experiments/e112/job_t2_export.sh`。它對三個認人網路和測試棋局的 12 個開局局面，內部呼叫 `e112_export.py --pset P12`。
- 推論：`bash scripts/experiments/e112/job_real_compute.sh`，也就是 `python3 scripts/experiments/e112/e112_real.py compute`，用 GPU，β=20。
- 輸入：`t2_pos_finh_{full,h0,h1}.npz`、`real_v7q_S_*.npz`。輸出：`.scratch/e112/final/real_X_{pub,priv}.npz`（400×600）。

#### 12. 候選池分類器（E123，600 類邏輯迴歸）

每個池各擬合一次，用 CPU，需要 scikit-learn 1.9.1。

```bash
PYTHONPATH=src:scripts:. .venv/bin/python scripts/experiments/e123/e123_make.py real
```

參數是 C=1.0、lbfgs。向量是小認人網路、大型認人網路和逐局面編碼器各自 L2 正規化後串接，再 L2、乘 8。輸入：`.scratch/{e90/r_h,e99/r,e86/r}/pe_test_{pub,priv}.npy` 和 `real_v7q_S_*.npz`。輸出：`.scratch/e123/final/real_Xlr_{pub,priv}.npz`。沒有權重檔，每次都在候選庫上現場擬合。

#### 13. 成對比對網路（E127，兩個：h0、h1，406,024 參數）

- 訓練資料：先 `.venv/bin/python scripts/experiments/e127/make_rows.py`，選訓練與留出玩家和棋局，寫出 `.scratch/e127/rows_*.npy`。再 GPU `bash scripts/experiments/e127/job_export.sh`，用兩個半數玩家的認人網路匯出逐局面向量 `pos_*_{val,trA,trB,test}.npz`。
- 訓練（GPU，兩個網路依序）：`bash scripts/experiments/e127/job_train.sh p1r cross "--steps 12000 --lr 1e-3 --lr_p 3e-4 --B 8 --M 24,58 --mon_every 1000 --ckpt_every 2000 --cand_sa 1 --qry_sa 1 --drop 0.1 --resid 1"`。本體是 `e127_train.py`，模型是 `e127_model.py`，資料是 `e127_data.py`。
- 推論（對真實測試題，GPU）：`bash scripts/experiments/e127/job_real_x.sh real`，由 `real_x.py` 執行，兩個網路各自對列標準化後取平均。
- 輸入：`pos_finh_h{0,1}_test.npz`、`sigma_finh_h{0,1}.npy`、`real_v7q_S_*.npz`。輸出：`.scratch/e127/p1r_finh_h{0,1}/last.pt`（權重）和 `.scratch/e127/final/real_Xp1r_{pub,priv}.npz`。

#### 14. 一對一先驗（CSLS，k=3、α=.5，無訓練）

函式是 `csls_qq`（`scripts/lib/e81_lib.py`），在最後一步內呼叫。E110 做了偽查詢驗證並選定 k，程式在 `scripts/experiments/e110/{e110_t2dump,e110_make,e110_an,e110_feats2}.py` 和 `bash scripts/experiments/e110/run_dump.sh`。

#### 15. 最終組合

見上面的「最後一步」。

```bash
.venv/bin/python final/make_player_final.py --work-dir <WORK_DIR> --out out/player_v7x.csv
```

算式是 c = rowz(e) + 0.5·X_e112 + 1.0·X_lr + 0.35·X_p1r，再做 CSLS，取前 5，最後與舊題 800 列拼接。原型是 `scripts/experiments/e127/make_final.py`。格式檢查用 `scripts/check_submission.py` 和 `scripts/oneoff/p1111_assemble.py`。

#### 16. 舊題管線（基底版，v6）

見上面「舊題 800 列」的七步表。

- 訓練：`scripts/predict.py train --task player --player-model e04v2 --include-holdout`，這是 E04v2 重排模型，程式在 `src/gogame/player_rerank.py`。
- 強度估計器特徵推論：`scripts/predict_se.py extract ...`，用外部權重。
- 融合與轉導解碼：`scripts/oneoff/e27_player_fusion.py fuse`。

### 本 repo 找不到程式的部分

- `.scratch/e40/fin/halves.npz`（深度度量編碼器的甲乙半分組）：原專案當時用一次性指令建立，沒有獨立腳本。分組邏輯等於 `scripts/oneoff/e39_final.py` 的 `groups_final` 加其餘玩家。
- Colab、學校和學長主機的工作排程與監看腳本（各平台的 `*_ctl.py`、`watch_*.sh`、`run_*.sh`）和個人機器設定：不公開。平台無關的執行本體都在上面的表裡。
- 舊題管線的訓練端強度估計器表（`se_uid.parquet`）與兩個度量檔：中間檔已刪（見「舊題 800 列」），抽取程式還在。

## 已知限制

1. 全流程重跑成本高。從棋譜重新編碼與重訓，至少約 76 小時 GPU。已知分項有認人網路三版 20.8、6.2、14.4 小時，大型認人網路全部玩家版約 25 小時，逐局面全部局面抽取約 9.2 小時。缺的分項沒有紀錄，所以實際更多，也沒有整條量過。HumanSL 掃描約 100 萬盤另計。
2. GPU 結果不會位元重現。GPU 上 `index_add_` 的順序不固定，又用 fp16，各平台的 torch 版本也不同。KataGo 掃描用 1 visit 加隨機對稱，重掃逐盤有雜訊。CPU 的最後組合步驟則是位元重現。
3. sklearn 必須是 1.9.1，LightGBM 是 4.7.0。
4. 上游中間檔不在 repo。腳本多半以 `.scratch/…` 為輸入輸出。各階段腳本是實驗過程中複製演進而來，每個階段都複製了前一階段的程式，沒有單一入口腳本能從棋譜整條重跑。
5. Player v7x 是未通過預先登記判準的備選。Public 池結構 a 的增益是 +.0028，門檻是 +.005，我們看過之後決定上傳。
6. 外部模型（HumanSL）的訓練棋譜來源，以及是否包含比賽棋局，作者沒有公開說明，無法排除。
7. 部分測試（約 13 個）需要官方資料，沒有資料時會被 skip，測試用棋譜是合成資料。
8. 舊題 800 列（v6）的最後一步無法重跑。缺 `metric_se.npz`、`metric_hsl.npz`（度量檔）和舊題 90,000 盤的 SE 特徵表，11/24 清理時已刪。基底 v3 可由官方資料重現（sha256 前 12 碼 `b51ae1671927`）。保存檔 `old_v6_800.csv`（`b4eed172f033…`）的 sha256 與格式可以核對。
9. `colab/reproduce_final.ipynb` 只在本機以 `jupyter nbconvert --execute` 驗證過，還沒有在真正的 Colab 上跑過。
10. 另一題見 `aicup2026-go-rank`。兩邊的共用程式（`src/gogame/` 與部分腳本）各放一份，彼此可能略有出入，只放該題用到的腳本。
