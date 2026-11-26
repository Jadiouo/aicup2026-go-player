# Player（競賽一）管線

最終上傳 **Player v7x**（Private 第 22 名 0.8836589）。提交檔 1,600 列 = 舊題 800 列 + 新題 800 列，兩半用不同方法：

- **舊題 800 列**（`pub_`/`priv_`，候選庫每人 100 盤黑棋）：v6（E04v2 reranker＋SE＋HSL 融合＋轉導解碼）。保存檔 `old_v6_800.csv`（sha256 `b4eed172f033…`，見 ASSETS.md）；執行順序、輸入輸出與缺件見下方「舊題 800 列（v6）」。
- **新題 800 列**（`pub2_`/`priv2_`，候選庫 600 人×每人 10 盤黑白混、查詢 1/2/3/5 盤）：v7a→v7x 一路疊加的管線（本頁重點）。

`eNN` 是實驗編號；腳本路徑 `scripts/experiments/eNN/` 與 `scripts/oneoff/eNN_*.py`。除另有註明，指令以 repo 根目錄為工作目錄、`PYTHONPATH=src:scripts:.`；腳本頂端 docstring 有完整參數。

## 管線（由下到上）

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

## 步驟

| # | 階段 | 腳本 | 輸入 → 輸出 | 硬體／耗時 |
|---|---|---|---|---|
| 1 | 資料處理：SGF、L0 | `src/gogame/{sgf,board,features_l0}.py`、`scripts/build_l0_cache.py`、`scripts/make_split.py`、`scripts/predict.py` | 棋譜 → L0 稠密／稀疏特徵；以 player_id 分組的切分 | CPU，分鐘～小時 |
| 2 | 特徵：HumanSL 掃描 | `scripts/katago/{scan_setup.sh,setup_katago.sh,scan_runner.py,scan_merge.py,scan_pins.json}`；測試端 `scripts/oneoff/p1111_hsl_prepare.py`、`scripts/experiments/e34/scan_runner_w2.py`（後半盤 W120） | SGF → 逐手 log p(落子)、名次、熵（11 個段位 profile，1 visit）→ `data/test_scan_p1111/hsl*_test_games.parquet` | GPU（KataGo CUDA）。訓練端約 100 萬盤：Colab L4 第 1 層 43.8 units（約 2.35 盤／秒）＋ 學校 RTX8000 第 2 層約 80 萬盤 |
| 3 | 特徵：KataGo 主幹向量 | `scripts/experiments/e13/{encode,train}.py`（`KataEncoder`、`load_katago_trunk`）、`scripts/oneoff/{e52_kata_trunk2,e56_trunk,e62_trunk,e69_trunk}.py` | SGF → 逐盤向量（落子點 384 ⊕ 全盤平均 384） | GPU，本機約 1,200 局面／秒 |
| 4 | 編碼：逐局面編碼器 pe86 | `scripts/oneoff/e77_enc.py`、`e80_{enc,extract,final}.py`、`e86_{prep,gpu,colab_prep,colab_enc,final}.py`、`e79_{items,prep_local,tok,gpu}.py` | 全部己方局面的主幹向量（PCA-64）→ 每盤 128 維 | GPU；全局面抽取 Colab L4 9.2 小時，編碼器每個 12.6 分鐘 |
| 5 | 訓練：S 認人網路 | `scripts/experiments/e89/{data_prep,train_s,fastboard,driver}.py`、`scripts/experiments/e90/{make_data_h,train_e90h,driver_e90h,assemble_pe_h}.py` | 約 100 萬盤（18 個 19×19 平面，10 區塊×128 通道 ResNet，cosine 分類 7,365 類）→ `finh_{full,h0,h1}/final_model.pt`、每盤向量 `pe_test_{pub,priv}.npy` | GPU；`finh_full` Colab L4 20.8 小時、`finh_h1` 14.4 小時、`finh_h0` 本機 5070 Ti 6.2 小時 |
| 6 | 訓練＋推論：度量、深度編碼器、booster | `scripts/oneoff/{e38,e39,e40,e41,e52,e62}_final.py`、`scripts/experiments/e90/{e90_final,e90_final_core,e90_feats2}.py` | 白化 LDA 度量、InfoNCE 深度度量、cross-fit；60 個模擬池×400 題×40 候選＝960,000 列訓練 booster（lambdarank，600 輪，seed 8）→ `real_v7q_S_{pub,priv}.npz`（e、top、pool、qids）＋`models/v7q_S.*` | CPU（booster 約 13 分鐘；度量／編碼器 GPU 各約 14 分鐘） |
| 7 | 推論：X_e112（局面核密度比對） | `scripts/experiments/e112/e112_{export,lib,stage2,real,amend1}.py` | 三個 S 網路的逐局面向量 → `real_X_{pub,priv}.npz`（400×600） | GPU 匯出＋CPU |
| 8 | 推論：X_lr（候選池分類頭） | `scripts/experiments/e123/e123_{lib,make,val}.py` | 每池 6,000 盤向量 → 600 類 sklearn LogisticRegression（C=1，lbfgs）→ `real_Xlr_{pub,priv}.npz` | CPU（**需 sklearn 1.9.1**） |
| 9 | 訓練＋推論：成對比對網路 X_p1r | `scripts/experiments/e127/{make_rows,e127_export,e127_data,e127_model,e127_train,e127_x,e127_eval0,e127_score}.py`、`real_x.py`、`job_export.sh`、`job_train.sh`、`job_x.sh`、`job_real_x.sh` | 逐局面向量匯出 → 訓練（12,000 步、lr 1e-3、seed 0）→ 對真實 test2 推論 → `real_Xp1r_{pub,priv}.npz` | GPU（本機 5070 Ti，0.04 s／步、VRAM 7.2 GB，每網路約 8 分鐘） |
| 10 | **最終組合** | **`final/make_player_final.py`**（原型：`scripts/experiments/e127/make_final.py`、`scripts/experiments/e123/e123_make.py`） | 上述 npz＋`old_v6_800.csv` → 1,600 列提交檔；並用 `scripts/oneoff/p1111_assemble.py` 驗證 | CPU，數秒 |

步驟 9 的訓練指令（原紀錄）：

```bash
bash scripts/experiments/e127/job_train.sh p1r cross "--steps 12000 --lr 1e-3 --lr_p 3e-4 --B 8 --M 24,58 --mon_every 1000 --ckpt_every 2000 --cand_sa 1 --qry_sa 1 --drop 0.1 --resid 1"
```

（`job_*.sh` 用系統 `python3`＋torch；原作者透過本機 GPU 排隊工具 `gpujob` 與 `systemd-run` 記憶體上限執行，這兩者不是必要的。）

**逐元件的訓練指令、推論指令、輸入、輸出見本頁最後一節「元件手冊」。**

權重與中間檔的大小、sha256 與放置位置見[`ASSETS.md`](ASSETS.md)；Colab 版最後一步見 `colab/reproduce_final.ipynb`。

## 最後一步：指令、輸入、輸出

```bash
.venv/bin/python final/make_player_final.py \
    --work-dir <WORK_DIR> --out out/player_v7x.csv [--w-new 0.35] [--reference 上傳檔.csv]
# 環境變數 DATA_DIR 指向含 contest_1111/ 的資料夾（驗證用）
```

| 輸入（`<WORK_DIR>` 下） | 內容 |
|---|---|
| `e90/final/make/real_v7q_S_{pub,priv}.npz` | `e`（400×40 booster 分數）、`top`（短名單）、`pool`（候選人 id）、`qids` |
| `e112/final/real_X_{pub,priv}.npz` | `X` 400×600 |
| `e123/final/real_Xlr_{pub,priv}.npz` | `X` 400×600 |
| `e127/final/real_Xp1r_{pub,priv}.npz` | `X` 400×600、`ng` |
| `e62/old_v6_800.csv` | 舊題 800 列（`--old800` 可改） |

輸出：`--out` 指定的 1,600 列 CSV（`question_id,top_1..top_5`）。`--w-new 0` 得到前一版 v7w（sha256 前綴 `21a6fde0`）。

## 舊題 800 列（v6）

提交檔前一半（`pub_`/`priv_`，候選庫每人 100 盤黑棋、查詢 5／10／15／20 盤各 100 題）= **v6**。2026-11-06 產生一次（sha256 `b4eed172f033c7a20431c439216a4678022342765615d79610a1851764e98d10`），之後 v7j2 起的每個版本都原樣沿用。
**狀態：保存檔可核對（`final/check_old_v6.py`）；由程式重算的最後一步目前跑不了，因為缺 3 個已被刪除的中間檔（下表）。**

方法：
1. 基底 v3 = E04v2 LightGBM lambdarank 重排（L0 特徵前 20 名，586 欄；`src/gogame/player_rerank.py`）＋轉導解碼。
2. 融合 e′ = e − 0.5·λ_se·D_se − 0.5·λ_hsl·D_hsl，λ_se = 0.5、λ_hsl = 0.25（實際係數 0.25、0.125）。D 為白化 LDA 空間的馬氏距離平方，再除以 (1/N_q + 1/n_lib)；度量只用 fit 玩家擬合。SE 用 `se_mean` 一維；HSL 每位候選人隨機取 20 盤（`default_rng(20)`）、查詢盤全取。
3. 轉導解碼：互為最佳且 softmax ≥ τ = 0.9 的配對先定案、移出後重算，3 輪後再做 Hungarian 一對一（`scripts/oneoff/player_sim3.py: trans_t5`）。

執行順序（`GOGAME_COLOR_SEMANTICS=actual`；指令以 repo 根為工作目錄、`PYTHONPATH=src:scripts:.`）：

| # | 指令 | 輸入 → 輸出 | 硬體／耗時 | 現況 |
|---|---|---|---|---|
| 1 | `python scripts/predict.py train --task player --player-model e04v2 --include-holdout`（需先做步驟 1 的切分與 L0 快取） | 訓練棋譜 → `data/models_final/player_e04v2.pkl`（含 H 重訓版） | CPU | 權重在 Release（ASSETS.md「自訓權重」） |
| 2 | `python scripts/oneoff/e27_player_fusion.py v3`（可設 `V3_REF=<v3 提交檔>` 逐列比對） | `data/contest_1104/*`＋步驟 1 的模型 → `.scratch/e27/v3_repro.csv` | CPU，約 1.5 分鐘 | **已在公開 repo 的副本實跑：sha256 前 12 碼 `b51ae1671927`，與 v3 上傳檔逐列相同（2026-11-26）** |
| 3 | `python scripts/oneoff/player_sim3.py prep` | 訓練端 SE 表 `.scratch/e21/se_uid.parquet`（`scripts/experiments/e21/extract.py`，GPU 12.9 小時）、`data/cache/e10/hsl_games.parquet`、`data/cache/games_meta.parquet`、切分 → `.scratch/player_sim3/metric_se.npz`、`metric_hsl.npz` | CPU（SE 表要 GPU） | **缺：兩個度量檔與 `se_uid.parquet` 已刪**（HSL 的來源表在；SE 的來源表不在） |
| 4 | `python scripts/oneoff/e27_player_fusion.py games` | `data/contest_1104/*` → `.scratch/e27/se_games.parquet`（90,000 盤）、`hsl_games_in.parquet`（26,000 盤） | CPU | 可重跑 |
| 5 | 把 `se_games.parquet` 切成 9 個 10,000 盤分塊 `.scratch/e27/in/c{0..8}.parquet`，逐塊 `python scripts/predict_se.py extract --games-parquet … --out-dir .scratch/e27/se_out/c{k} --model <se_go.pt> --device cuda --half`；再 `python scripts/oneoff/e27_merge_se.py` | → `.scratch/e27/se/se_test_games.parquet`（`uid,se_mean,se_std,se_n`） | GPU，每塊約 7 分鐘（RTX8000，合計約 1 小時） | **缺：此表已刪**；可重抽（`colab/full_pipeline.ipynb` 段 9 有程式） |
| 6 | `python scripts/oneoff/e27_hsl_prepare.py data/test_scan_e27player` → `scripts/katago/scan_runner.py`（分片可續跑）→ 入帳 → `python scripts/predict_hsl.py --root data/test_scan_e27player features` → 複製 `hsl_test_games.parquet`、`features_manifest.json` 到 `.scratch/e27/hsl/` | 26,000 盤 → HSL 特徵表（104 欄＋uid） | GPU，約 4 小時（RTX8000，1.9 盤／秒） | 特徵表是官方資料衍生中間檔，僅提供主辦方（ASSETS.md 的 `e27player_hsl_test_games.parquet`，14.4 MB） |
| 7 | `python scripts/oneoff/e27_player_fusion.py fuse [<含 se/、hsl/ 的資料夾> <輸出 CSV>]` | 步驟 3、5、6 的輸出＋步驟 1 的模型 → 800 列 CSV（＝ `old_v6_800.csv`）＋`fuse_summary.json` | CPU，約 1.5 分鐘 | **目前無法執行**（缺步驟 3、5 的檔） |

保存檔與紀錄的一致性（可在無上游檔下核對）：

```bash
python final/check_old_v6.py <old_v6_800.csv> [--v3 <v3 提交檔>]
# sha256 前 12 碼 b4eed172f033；800 列格式；與 v3 相比 top-1 變 10 題、任一名次變 795 題（融合當時紀錄的數字，依 N=5/10/15/20 為 6/2/1/1 題 top-1 改變）
```
2026-11-26 實測：sha256 相符、格式 0 錯誤、與 v3 的差異數與紀錄相同（10／795）。

其他注意：
- 基底模型用含 H 玩家重訓的 `player_e04v2.pkl`，λ 是用不含 H 的模擬池選的（`tune_combo_table.csv` 中 tune 分數最高者），測試端沒有再調；離線模擬 dev 40 池 .9927 → .9967（Δ +.0040），模擬與真實 Public 差約 .007（舊題那一半的 Public 約 1.000，為推論值，非實測）。
- 新題那一半**不**用這個轉導解碼（新題結構未知，轉導／Hungarian 在「同人多題」結構大幅變差）。
- SE 輸入編碼 `scripts/experiments/e21/se_encode.py` 與 SE 權重授權的說明見README「授權」。

## 驗證設計

一律以 player_id 分組切分（同一玩家不同時出現在訓練與評估）。新題沒有答案，離線用「真實候選庫偽查詢」（每人 10 盤中 6 盤當庫、4 盤當查詢）與 16 個模擬池評量。

## 元件手冊：每個元件的訓練、推論、輸入、輸出

上面的「步驟」表是由下到上的流程；這一節是逐元件的指令。慣例：

- 指令以 repo 根目錄為工作目錄；CPU 腳本用 `PYTHONPATH=src:scripts:. .venv/bin/python`，標「GPU」的用系統 `python3`＋torch（`requirements-gpu.txt`）。完整參數看各腳本頂端 docstring；`.scratch/`、`data/` 的位置見 README「路徑約定」。
- 「報告名稱」是報告正文用的描述性名稱；「內部代號」是程式與紀錄裡的編號。
- 原作者用本機 GPU 排隊工具與 `systemd-run` 記憶體上限包住重工作，這兩者不是必要的。

### 資料與特徵（多半無訓練）

**1. 手工特徵（L0）** — 無訓練。
- 抽取：`.venv/bin/python scripts/build_l0_cache.py [--jobs 14] [--chunk 20000]`；測試端由 `scripts/oneoff/e38_final.py predict` 即時呼叫 `src/gogame/features_l0.py`，結果存 `.scratch/p1111/z_t2/Z_{pub,priv}.npz`。
- 輸入：官方棋譜（`data/training*`、`data/contest_1111/`）。輸出：`data/cache/l0/{dense.parquet,sparse.npz}`。
- 官方測試題的欄位與 id 整理：`.venv/bin/python scripts/oneoff/p1111_inspect.py --data-dir data/contest_1111`（寫 `.scratch/p1111/normalized/`）。

**2. 訓練資料包（E29 bundle）** — 無訓練。`.venv/bin/python -m scripts.experiments.e29.prep_bundle` → `.scratch/e29/bundle/{X.npy,aux.npz,manifest.json}`（非保留集玩家的逐盤特徵與折角色；後面的 `e36`、`e40_finalprep`、`e55_prep` 都讀它）。

**3. HumanSL 落子統計（前 60 手／前 120 手）** — 外部 KataGo HumanSL，無訓練。
- 抽取（訓練端）：`bash scripts/katago/scan_setup.sh`（下載並以 sha256 釘選 KataGo 與網路）→ `scripts/katago/scan_runner.py`（前 60 手）／`scripts/experiments/e34/scan_runner_w2.py`（第 61–120 手），參數見兩者 docstring；合成訓練端 W120 表：`python -m scripts.experiments.e36.build_w120_full old|new|train` → `.scratch/e36/bundle/X120_full.npy`。
- 抽取（測試端）：`GOGAME_COLOR_SEMANTICS=actual .venv/bin/python scripts/oneoff/p1111_hsl_prepare.py data/test_scan_p1111` → `scripts/predict_hsl.py --root data/test_scan_p1111 run --backend school ...` → `scripts/predict_hsl.py --root ... features`；後半盤合併：`.venv/bin/python scripts/oneoff/e41_late_merge.py` → `data/test_scan_p1111/hsl120_test_games.parquet`（uid＋104 欄）。
- 輸入：SGF。輸出：逐手機率、名次、熵 → 逐盤 104 欄。

**4. KataGo 主幹向量（逐盤）** — 外部預訓練網路的內部啟動值，無訓練。E52→E56→E62→E69 四代，每代「CPU 準備局面＋GPU 抽向量」：
- 前半盤、6 局面（E52）：`.venv/bin/python scripts/oneoff/e52_kata_prep3.py`；GPU `E52_KDIR=kata3 python3 scripts/oneoff/e52_kata_trunk2.py`。
- 前半盤、12 局面、多層（E56）：`.venv/bin/python scripts/oneoff/e56_prep.py --set final`；GPU `python3 scripts/oneoff/e56_trunk.py --set final`。
- 後半盤（E62）：`e62_prep.py --set final`；GPU `e62_trunk.py --set final`。
- 多區塊 3/6/12/15（E69）：`e69_prep.py --set final`；GPU `e69_trunk.py --set final`。
- 輸入：SGF、KataGo `b18c384nbt-humanv0.ckpt`（外部，見 ASSETS.md）。輸出：`.scratch/e52/kata3/`、`.scratch/e56/final/`、`.scratch/e62/final/`、`.scratch/e69/final/` 下的 `blk{i}_gv.npy`（逐盤向量）。編碼器 `scripts/experiments/e13/encode.py`。

### 神經網路

**5. 逐局面棋風編碼器（pe86；內部代號 E77→E80→E86）** — 訓練（6 個：全部／甲半／乙半，各有「模擬池」與「最終」兩組；最終用 3 個）。
- 準備：`python scripts/oneoff/e77_prep_full.py --set pca`（PCA 擬合局面）→ GPU `python3 scripts/oneoff/e77_gpu_full.py --sets pca`（寫 `.scratch/e77/pca_full.npz`，PCA 96 維；E80／E86 取前 64 維）；E79 前後視窗 24 個 token：`e79_items.py`、`e79_prep_local.py`、GPU `e79_gpu.py`、`e79_tok.py`；`e80_meta.py`、`e80_prep.py`、GPU `e80_extract.py`；E86 全局面：`e86_items.py`、`e86_meta.py`，再 `e86_prep.py STAGE ITEMS.pkl`＋GPU `e86_gpu.py STAGE`，或 Colab 版 `e86_colab_prep.py`。
- 訓練＋匯出（最終）：`python scripts/oneoff/e86_colab_enc.py [--only sim,final]`（一次訓練六個編碼器並匯出）；較早的 E80 版 `python3 scripts/oneoff/e80_enc.py sim|final [--only h0,h1,full]`（24 個 token）；編碼器定義與訓練函式 `scripts/oneoff/e77_enc.py`。
- 推論：同上腳本匯出 `pe_final_h0.npy`、`pe_final_h1.npy`（訓練池用）與 `pe_test_{pub,priv}.npy`（測試用）。
- 輸入：全部己方局面的主幹向量（PCA-64）。輸出：每盤 128 維單位向量。權重：`enc_*.pt`（已不在，只剩向量；見 ASSETS.md）。

**6. 認人網路（S；內部代號 E89→E90，三個：`finh_full`、`finh_h0`、`finh_h1`）** — 10 區塊×128 通道 ResNet。
- 資料：`PYTHONPATH=src:. .venv/bin/python scripts/experiments/e90/make_data.py union|test|meta`（`u_games.npz`、`rosters.npz`、`t_games.npz`、`export_spec*`）→ `.venv/bin/python scripts/experiments/e90/make_data_h.py build`，再 `python scripts/experiments/e90/make_data_h.py concat [--dir DIR]`（加入保留集的盤，得 `uh_games.npz`、`rosters_h.npz`、`export_spec_h.*`）。自寫棋盤引擎 `scripts/experiments/e89/fastboard.py`（numba）。
- 訓練（GPU，可續跑）：`python scripts/experiments/e90/driver_e90h.py --roster finh_full --root DIR --accum 2 --workers 8 --export auto`（`finh_h0`、`finh_h1` 同理；DIR 內放 `uh_games.npz rosters_h.npz t_games.npz export_spec_h.json export_spec_h.npz`）。驅動器呼叫 `train_e90h.py`（20 輪、種子 0）。
- 推論（逐盤向量）：訓練結束時 `--export auto` 一併寫 `vec_*.npz`；整理成 booster 用的檔：`python scripts/experiments/e90/assemble_pe_h.py DIR .scratch/e90/r_h` → `pe_final_h0.npy`、`pe_final_h1.npy`、`pe_test_{pub,priv}.npy`。逐局面向量匯出另見元件 9。
- 輸入：約 100 萬盤的己方局面 18 個 19×19 平面。輸出：`finh_*/final_model.pt`（權重）、每盤 128 維向量。

**7. 大型認人網路（E99，16 區塊×224 通道，三個）** — 只當候選池分類器的向量來源，不進重排模型。
- 訓練（GPU，可續跑；與元件 6 同資料）：`python scripts/experiments/e99/driver_e99.py --roster finh_full --root DIR --epochs 12`（`finh_h0`、`finh_h1` 同理；DIR 內同元件 6 的 `uh_games.npz`、`rosters_h.npz`、`export_spec_h.*`；每網路 `--accum 8`、種子 0 由驅動器固定）。訓練本體 `scripts/experiments/e99/train_e99.py`（元件 6 的 `train_e90h.py` 的複本，只改規模）。
- 推論（逐盤向量）：訓練時 `--export auto` 寫 `vec_*.npz`；`python scripts/experiments/e99/assemble_pe_e99.py [NETS_DIR] [OUT_DIR]`（預設 `.scratch/e99/nets` → `.scratch/e99/r`）。
- 輸入：同元件 6。輸出：`.scratch/e99/nets/finh_full/final_model.pt`（Release 檔名 `finh_bigS_full__final_model.pt`）、`.scratch/e99/r/pe_test_{pub,priv}.npy`。
- 另：`e99_final.py` 是「用大網路取代／加入小網路」重排模型（v7r，兩版都沒贏、未上傳）的產檔程式，`test_feats3.py` 是它的特徵測試；最終管線不用。

**8. 深度度量編碼器（E40／E41，兩個 MLP 編碼器×全部／甲半／乙半）** — episodic 對比學習。
- 準備：`.venv/bin/python scripts/oneoff/e40_finalprep.py`（最終 Z 空間與標準化後的 HumanSL；Z 若被清掉用 `e41_finz.py` 重建）→ `e40_fin_inputs.py`、`e41_fin_inputs120.py`（測試端輸入）。
- 訓練＋嵌入（GPU）：`python3 scripts/oneoff/e40_deep_mlp.py --data fin --hsl 1 --out fin_full --half -1`（前 60 手 HumanSL 版）、`python3 scripts/oneoff/e41_deep_fin.py --data fin --hsl 2 --out fin120_full --half -1`（前 120 手版）；`--half 0|1` 訓練甲半／乙半版（設定：30,000 步、hidden 4096、P=512、dim 256，見 E40／E41 紀錄）。兩個編碼器串接除以 √2 得集成嵌入：`python scripts/oneoff/e50_ens.py --w60-dir .scratch/e40 --w60-tag fin --out finens --out-dir .scratch/e41`。
- 輸入：L0 Z＋標準化 HumanSL。輸出：`enc_*.pt`、`emb_finens_{h0,h1}_cap.npy`、`emb_finens_full_test_{pub,priv}.npy`。注意：甲乙半分組檔 `.scratch/e40/fin/halves.npz` 在原專案是當時一次性指令建立的，沒有獨立腳本。

### 統計模型與重排

**9. 白化線性判別度量（LDA 度量）＋召回融合** — 無神經網路；在重排模型訓練時擬合（同一支腳本，見元件 10）。L0 LDA（150 維、ridge 3e-2）、HumanSL LDA（15 維）、主幹向量 LDA；以馬氏距離的 z 分數和取前 K=40 位候選人。程式：`scripts/oneoff/e38_final.py`、`e39_final.py`（交叉擬合分組）。

**10. 重排模型（LightGBM lambdarank，v7q；792 特徵）** — 訓練＋推論在同一支腳本。
- 訓練＋測試端預測：`PYTHONPATH=src:scripts:. .venv/bin/python scripts/experiments/e90/e90_final.py make --s-dir .scratch/e90/r_h`（經 `e90_final_core.py` 使用 `scripts/oneoff/e86_final.py` 的訓練與預測碼：60 個模擬池×400 題×40 候選人＝96 萬列、600 輪、種子 8；Public／Private 池各自預測）。`e90_final.py preflight --s-dir ...` 先做不吃記憶體的檢查。
- 輸入：元件 1、3、4、5、6、8 的輸出；`.scratch/e90/r_h`（認人網路向量）；`.scratch/e86/r`（逐局面編碼器向量）。輸出：`.scratch/e90/final/make/real_v7q_S_{pub,priv}.npz`（`e`、`top`、`pool`、`qids`）、`models/v7q_S.{txt,json}` 與 5 個度量 npz。

**11. 局面對局面比對（E112，核密度；無訓練）**
- 匯出逐局面向量（GPU）：`bash scripts/experiments/e112/job_t2_export.sh`（三個認人網路 × 測試棋局的 12 個開局局面，內部呼叫 `e112_export.py --pset P12`）。
- 推論：`bash scripts/experiments/e112/job_real_compute.sh`（＝ `python3 scripts/experiments/e112/e112_real.py compute`，GPU，β=20）。
- 輸入：`t2_pos_finh_{full,h0,h1}.npz`、`real_v7q_S_*.npz`。輸出：`.scratch/e112/final/real_X_{pub,priv}.npz`（400×600）。

**12. 候選池分類器（E123，600 類邏輯迴歸）**
- 訓練＋推論（每個池各擬合一次，CPU，需 scikit-learn 1.9.1）：`PYTHONPATH=src:scripts:. .venv/bin/python scripts/experiments/e123/e123_make.py real`（C=1.0、lbfgs；向量＝小認人網路＋大型認人網路＋逐局面編碼器各自 L2 後串接再 L2、×8）。
- 輸入：`.scratch/{e90/r_h,e99/r,e86/r}/pe_test_{pub,priv}.npy`、`real_v7q_S_*.npz`。輸出：`.scratch/e123/final/real_Xlr_{pub,priv}.npz`。無權重檔（每次在候選庫上現場擬合）。

**13. 成對比對網路（E127，兩個：h0、h1；406,024 參數）**
- 訓練資料：`.venv/bin/python scripts/experiments/e127/make_rows.py`（選訓練／留出玩家與棋局，寫 `.scratch/e127/rows_*.npy`）→ GPU `bash scripts/experiments/e127/job_export.sh`（用兩個半數玩家認人網路匯出逐局面向量 `pos_*_{val,trA,trB,test}.npz`）。
- 訓練（GPU，兩網路依序）：`bash scripts/experiments/e127/job_train.sh p1r cross "--steps 12000 --lr 1e-3 --lr_p 3e-4 --B 8 --M 24,58 --mon_every 1000 --ckpt_every 2000 --cand_sa 1 --qry_sa 1 --drop 0.1 --resid 1"`（本體 `e127_train.py`、模型 `e127_model.py`、資料 `e127_data.py`）。
- 推論（對真實測試題，GPU）：`bash scripts/experiments/e127/job_real_x.sh real`（`real_x.py`；兩網路各自列標準化後平均）。
- 輸入：`pos_finh_h{0,1}_test.npz`、`sigma_finh_h{0,1}.npy`、`real_v7q_S_*.npz`。輸出：`.scratch/e127/p1r_finh_h{0,1}/last.pt`（權重）、`.scratch/e127/final/real_Xp1r_{pub,priv}.npz`。

**14. 一對一先驗（CSLS，k=3、α=.5；無訓練）** — 函式 `csls_qq`（`scripts/lib/e81_lib.py`），在最後一步內呼叫；E110 的偽查詢驗證與 k 的選擇：`scripts/experiments/e110/{e110_t2dump,e110_make,e110_an,e110_feats2}.py`、`bash scripts/experiments/e110/run_dump.sh`。

**15. 最終組合** — 見上節「最後一步」：`.venv/bin/python final/make_player_final.py --work-dir <WORK_DIR> --out out/player_v7x.csv`（c = rowz(e) + 0.5·X_e112 + 1.0·X_lr + 0.35·X_p1r → CSLS → 前 5 → 與舊題 800 列拼接）；原型 `scripts/experiments/e127/make_final.py`；格式檢查 `scripts/check_submission.py`、`scripts/oneoff/p1111_assemble.py`。

**16. 舊題管線（基底版，v6）** — 見上節「舊題 800 列（v6）」的七步表：訓練 `scripts/predict.py train --task player --player-model e04v2 --include-holdout`（E04v2 重排模型，`src/gogame/player_rerank.py`）；強度估計器特徵推論 `scripts/predict_se.py extract ...`（外部權重）；融合與轉導解碼 `scripts/oneoff/e27_player_fusion.py fuse`。

### 本 repo 找不到程式的部分

- `.scratch/e40/fin/halves.npz`（深度度量編碼器的甲乙半分組）：原專案當時以一次性指令建立，沒有獨立腳本（分組邏輯＝`scripts/oneoff/e39_final.py` 的 `groups_final`＋其餘玩家）。
- Colab／學校／學長主機的工作排程與監看腳本（各平台的 `*_ctl.py`、`watch_*.sh`、`run_*.sh`）與個人機器設定：不公開；平台無關的執行本體都在上表。
- 舊題管線的訓練端強度估計器表（`se_uid.parquet`）與兩個度量檔：中間檔已刪（見「舊題 800 列」），抽取程式在。
