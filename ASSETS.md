# 檔案清單（ASSETS）：Player（競賽一）

2026-11-26 在原專案實測（大小與 sha256 由腳本計算）。`colab/assets_manifest.json` 是同一份清單的機器可讀版（`category` 欄分類）；`colab/reproduce_final.ipynb` 依它下載、驗 sha256。
本 repo **不含**下列任何檔案。每個檔案屬於三類之一：

| 類別 | 做法 |
|---|---|
| **自訓權重（公開 Release）** | 我們自己訓練出來的模型參數。放在這個 repo 的 GitHub Release，下載網址 `RELEASE_URL`：**待補**（Release 建好後填入 `colab/reproduce_final.ipynb` 與 README） |
| **官方資料衍生中間檔（僅提供主辦方）** | 由主辦方的官方棋譜算出來的特徵、向量、各模型對測試題／驗證題的機率與預測檔。**不放 repo、不放 Release**；可應主辦單位要求以私人連結另行提供。沒有它們時如何重新產生，見 README |
| **外部（不散布，附下載網址）** | KataGo 執行檔與網路、HumanSL 檢查點、Strength Estimator 權重。我們不再散布，只給官方網址與放置路徑（第 3 節） |

分類依據：只含「模型參數」（權重、投影矩陣、迴歸係數）者為自訓權重；凡含官方棋局的內容、由官方棋局逐盤算出的數值、對官方題目的預測，或內含官方玩家 id／標籤者，一律列為中間檔。
**Release 上的檔名**：Release 沒有資料夾，清單的「資產名」就是 Release 上的檔名（`/` 已換成 `__`）。**放回 repo 的位置**是相對 repo 根（`data/`、`.scratch/` 約定見 README）。

## 1. 自訓權重（公開 Release）

| 資產名 | 用途 | bytes | sha256 | 放回 repo 的位置 | 最後一步需要 |
|---|---|---:|---|---|:---:|
| `player_e04v2.pkl` | E04v2 reranker（含 H 重訓版） | 795,221 | `8c9517ecd5ecfa7c07b44f59058a87c5b484c78801a3e3d4f1e44f78caab87e3` | `data/models_final/player_e04v2.pkl` | 否 |
| `player_e04v2.txt` | 同上的 LightGBM 文字檔 | 768,457 | `3b6d0f552d922aa54f78791cc140d331e67658d9173fe3e713850270f861b50f` | `data/models_final/player_e04v2.txt` | 否 |
| `finh_full__final_model.pt` | S 認人網路（測試端；7,365 類） | 15,930,521 | `fec071cad0cabc7aacbecbb81297dfc3ac87ca51e1b73bcbb01b0828252b2a59` | `.scratch/e90/colab/finh_full/final_model.pt` | 否 |
| `finh_h0__final_model.pt` | S 認人網路（交叉擬合 h0） | 14,174,873 | `92ed8dd4a24ac3fc21d7bb69011ef3183d2ba2d2a76b88525e317d9934470eec` | `.scratch/e90/colab/finh_h0/final_model.pt` | 否 |
| `finh_h1__final_model.pt` | S 認人網路（交叉擬合 h1） | 14,185,817 | `59d95369e73e352ab5bd37108194f2af9db80ba85cfb79b2fecd81a3ed7cad5e` | `.scratch/e90/colab/finh_h1/final_model.pt` | 否 |
| `p1r_finh_h0__last.pt` | Player 成對比對網路 h0（E127） | 4,928,727 | `dc1e81a3fd47d580d59c97c4004c30ecc03a8a9f6b1277b6c96e08e001275814` | `.scratch/e127/p1r_finh_h0/last.pt` | 否 |
| `p1r_finh_h1__last.pt` | Player 成對比對網路 h1 | 4,928,727 | `ff31fa32ca12ad3310b2d0c59b089fdb08ae4277c741c63e48303398a1ea8afc` | `.scratch/e127/p1r_finh_h1/last.pt` | 否 |
| `sigma_finh_h0.npy` | E127 白化用逐維 std（h0） | 640 | `c2d4e033106b1d1df13acc3020b35fbabb56a42c499ef6ac35fc4264b6403ecb` | `.scratch/e127/sigma_finh_h0.npy` | 否 |
| `sigma_finh_h1.npy` | 同上（h1） | 640 | `12fe6e285f2355a9ea66db0094c6055937b20c3cb05acd10f26659992e0e22ad` | `.scratch/e127/sigma_finh_h1.npy` | 否 |
| `v7q_S.txt` | Player 最終 LightGBM booster（792 特徵） | 1,139,907 | `a51107e17bd7baab91e1cefb2ec7062d948adec460d1d9b67154f3e38c861297` | `.scratch/e90/final/make/models/v7q_S.txt` | 否 |
| `v7q_S.json` | booster 設定 | 430 | `76cb81f7a33e0235cc45e4f7242ec55b07b88827594fc8bf88513d523b7641d8` | `.scratch/e90/final/make/models/v7q_S.json` | 否 |
| `v7q_S_metric.npz` | booster 用度量 | 1,334,402 | `ab00d7dd5b77936b4d88d4a371dcceb4805fe2e251b21076e6823a4aa6c8f295` | `.scratch/e90/final/make/models/v7q_S_metric.npz` | 否 |
| `v7q_S_blk.npz` | booster 用分塊度量 | 2,326,254 | `a019653dec5c9883d29d8fa35ad8bd384a61dc70debf440c70cb8bade9ff4aa7` | `.scratch/e90/final/make/models/v7q_S_blk.npz` | 否 |
| `v7q_S_late.npz` | booster 用後半盤度量 | 968,644 | `4a57302e6be359c098eb8c128f17d5523c020f913a841cfc252ddf809bea90ed` | `.scratch/e90/final/make/models/v7q_S_late.npz` | 否 |
| `v7q_S_trk.npz` | booster 用主幹度量 | 1,583,044 | `81bcf4144b95168525a89814317468ce158f90a5ce9516eeb6dd8e0c292c2060` | `.scratch/e90/final/make/models/v7q_S_trk.npz` | 否 |
| `v7q_S_trkhalf.npz` | booster 用後半盤主幹度量 | 3,172,322 | `a6e099a8080d2f4aa66cc70866afa90348b35c7b4642c070b049039101a5d021` | `.scratch/e90/final/make/models/v7q_S_trkhalf.npz` | 否 |
| `finh_bigS_full__final_model.pt` | E99 大網路 S（16 層×224）finh_full，只當 X_lr 的向量來源（原 ASSETS 未列，本輪補） | 62,196,265 | `c6c5e6f9d2595477ae4b515c0178aca70769d9dc4b451a88e330ff02b0a56798` | `.scratch/e99/nets/finh_full/final_model.pt` | 否 |

小計：17 檔，128,434,891 bytes（128.4 MB）；最大單檔 `finh_bigS_full__final_model.pt`（62,196,265 bytes）。沒有單檔超過 GitHub Release 的 2 GB 上限。

**公開前的處理**：`v7q_S.json` 原檔的訓練參數（`args`／設定）裡記有作者本機或遠端主機的絕對路徑；公開版已把這些字串替換掉（帳號目錄前綴 → `<home>`、專案根目錄前綴移除），**所有張量與數值逐項相同**（上傳前以程式逐一比對）。因此這幾個檔的 sha256 與作者內部紀錄的不同，**以本表為準**。

**注意**
- `.pkl` 檔（pickle）載入時會執行程式碼；請只使用本 repo Release 的檔案，並先核對 sha256。
- 以 KataGo HumanSL 檢查點微調的權重（Rank 的 `final_s0`、`ft4_s0`）屬於衍生自 HumanSL 網路的模型，使用時請一併遵守 KataGo Neural Network License（見 README「外部資源」）。

## 2. 官方資料衍生中間檔（僅提供主辦方；不在 repo、不在 Release）

| 資產名 | 用途 | bytes | sha256 | 放回 repo 的位置 | 最後一步需要 |
|---|---|---:|---|---|:---:|
| `player__real_v7q_S_pub.npz` | Player：booster 分數 e、短名單 top、候選人 id（Public 池） | 2,016,603 | `6126bf56cd2e86b02cc324f758f1859476bebc12ca67e99a8209a41ddcd74c5e` | `.scratch/e90/final/make/real_v7q_S_pub.npz` | 是 |
| `player__real_v7q_S_priv.npz` | Player：同上（Private 池） | 2,016,892 | `3695b13377e95778c34dcb9e50a2e48e1bde38243a9a892e412d81ea53c76b2f` | `.scratch/e90/final/make/real_v7q_S_priv.npz` | 是 |
| `player__real_X_pub.npz` | Player：X_e112（400×600） | 960,604 | `fb4f7809ccd4ccb59edf0fbe526e0f2330ec68d3f3212689830412444d714381` | `.scratch/e112/final/real_X_pub.npz` | 是 |
| `player__real_X_priv.npz` | Player：X_e112 | 960,604 | `abe3d2b1651d9c22752f6e6c00d2c49111358fbd23476665f87c5b9b8e6ea696` | `.scratch/e112/final/real_X_priv.npz` | 是 |
| `player__real_Xlr_pub.npz` | Player：X_lr（400×600） | 960,256 | `e361d49208639c93643dbb4c5e5df50718f0672c9612a9657dd8e45cddf3f939` | `.scratch/e123/final/real_Xlr_pub.npz` | 是 |
| `player__real_Xlr_priv.npz` | Player：X_lr | 960,256 | `314b278103de4c8cdb7caa5bbfb9301180664a4bf3adba2059d35c373095d1ee` | `.scratch/e123/final/real_Xlr_priv.npz` | 是 |
| `player__real_Xp1r_pub.npz` | Player：X_p1r（400×600） | 1,091,930 | `16ae9ff320f071cc4e3864b51e27c67724912c72c8d42cf7041901f296f52599` | `.scratch/e127/final/real_Xp1r_pub.npz` | 是 |
| `player__real_Xp1r_priv.npz` | Player：X_p1r | 1,091,930 | `6391db057321e96324f905486906a469c99083a31be68498aba1ea1833bb8622` | `.scratch/e127/final/real_Xp1r_priv.npz` | 是 |
| `player__old_v6_800.csv` | Player：舊題 800 列（v6），原樣沿用 | 55,242 | `b4eed172f033c7a20431c439216a4678022342765615d79610a1851764e98d10` | `.scratch/e62/old_v6_800.csv` | 是 |
| `ref__player_v7x.csv` | Player 最終上傳檔 | 115,242 | `b4ad40c24b5c6e991cc69bb08af060de23b3f68c96e64f5d7d79c36ca4c04643` | `ref/player_v7x.csv` | 否 |
| `ref__player_v7w.csv` | Player 前一版（--w-new 0 的比對目標） | 115,242 | `21a6fde0813d452d4974be5051cafaf44cb264d6b8c10367c3d4afc7c78262e6` | `ref/player_v7w.csv` | 否 |
| `e27player_hsl_test_games.parquet` | 舊題 26,000 盤的 HSL 特徵表（v6 融合輸入） | 14,433,911 | `d2cd1c51e411aee0238de198fe387d5e60da8aa4972eeafc9b88498bfad68b85` | `data/test_scan_e27player/hsl_test_games.parquet` | 否 |
| `e27_v3_repro.csv` | v3 重現檔（sha b51ae1671927…，v6 的基底） | 55,242 | `b51ae1671927ad26e170b746d76dd1d13edbf9b7d560b6b0e75a0edcaf84f00f` | `.scratch/e27/v3_repro.csv` | 否 |
| `player_up__pos_finh_h0_test.npz` | Player E127 推論輸入：逐局面向量（h0） | 226,709,864 | `d570222b1eff6d1fcd7b7a3da4834d6d2110c3e07f29313ba99379004ea67d99` | `.scratch/e127/pos_finh_h0_test.npz` | 否 |
| `player_up__pos_finh_h1_test.npz` | 同上（h1） | 226,709,864 | `aad298f1615004009884730c4db7d43bc7035091ac85576a7b34fb6d49429d61` | `.scratch/e127/pos_finh_h1_test.npz` | 否 |
| `player_up__t2_pos_finh_full.npz` | Player E112 輸入：逐局面向量（full） | 58,903,568 | `6bdb377cb56d2f61137cbb0278bcd175582e0c573e3a01f3e9568ade70f29088` | `.scratch/e112/t2_pos_finh_full.npz` | 否 |
| `player_up__t2_pos_finh_h0.npz` | 同上（h0） | 58,903,568 | `7be9dd66f90a687899235261e1d854cd3178fb5936385a7bc645f68f77fee0b6` | `.scratch/e112/t2_pos_finh_h0.npz` | 否 |
| `player_up__t2_pos_finh_h1.npz` | 同上（h1） | 58,903,568 | `953c8bb0c9671e4fc23299ad0980c4e79c2bf37a90f41bba038d7ef01729ea43` | `.scratch/e112/t2_pos_finh_h1.npz` | 否 |
| `player_up__pe_test_pub_small.npy` | 小 S 每盤向量（Public） | 3,635,328 | `60826a85553a2f254d19036d320e4d6e1dbe9f81516530ee99539f4ce0fc21f8` | `.scratch/e90/r_h/pe_test_pub.npy` | 否 |
| `player_up__pe_test_priv_small.npy` | 小 S 每盤向量（Private） | 3,635,328 | `dfe77bf0d63117bc5a381aca4c679c0f752fd99108ef4861543a23f0e82beef9` | `.scratch/e90/r_h/pe_test_priv.npy` | 否 |
| `player_up__pe_test_pub_bigS.npy` | 大 S（E99）每盤向量（Public） | 3,635,328 | `b97b2bedd3f24bf46d9acda2c0493548fb8328e0f25dde025c5ac79c47978e29` | `.scratch/e99/r/pe_test_pub.npy` | 否 |
| `player_up__pe_test_priv_bigS.npy` | 大 S（E99）每盤向量（Private） | 3,635,328 | `cb01afc278d8680c55973018e7f4a3c33f02a5da6dcb79a1a21ee68b6c1e404c` | `.scratch/e99/r/pe_test_priv.npy` | 否 |
| `player_up__pe_test_pub_pe86.npy` | pe86 每盤向量（Public） | 3,635,328 | `0e91fffe01e8b0486862b68620a742e5ad0c9a0107c7ea860670de52c63bd958` | `.scratch/e86/r/pe_test_pub.npy` | 否 |
| `player_up__pe_test_priv_pe86.npy` | pe86 每盤向量（Private） | 3,635,328 | `cc3ca73f536ac86624e463e968f243850e1103b6dd4cab1cf5416bc31d1cd146` | `.scratch/e86/r/pe_test_priv.npy` | 否 |
| `player_up__Z_priv.npz` | 新題 L0 特徵 Z（Private） | 59,384,896 | `cea6d154cde9e725d74ecf659791aeaac1c8ff732725da649e4c246431bb335d` | `.scratch/p1111/z_t2/Z_priv.npz` | 否 |
| `player_up__Z_pub.npz` | 新題 L0 特徵 Z（Public） | 59,328,096 | `dd5b7b391131214e0f7677be61ebde0284fe3d3adef5eb899b863d2c7ffd430e` | `.scratch/p1111/z_t2/Z_pub.npz` | 否 |

小計：26 檔，795,489,346 bytes（795.5 MB）；其中最後一步需要的 9 檔共 10,114,317 bytes（10.1 MB）。最大單檔 `player_up__pos_finh_h0_test.npz`（226,709,864 bytes）。

**取得方式**：這些檔案只提供給主辦方（私人連結）。放進同一個資料夾後，把 `INTERMEDIATE_DIR` 指到該資料夾；`colab/colab_utils.py` 的 `fetch_intermediate` 會依「資產名」或「放回 repo 的位置」找檔、複製到 repo、驗 sha256。

## 3. 外部（不散布；官方下載網址與放置路徑）

| 資源 | 官方下載網址 | bytes | sha256 | 放置路徑（相對 repo 根） | 授權 |
|---|---|---:|---|---|---|
| KataGo 執行檔 v1.18.2 | https://github.com/lightvector/KataGo/releases/download/v1.18.2/katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip | 51,350,638 | `2cb197fc9bc2050f29112abb5b83cdbad63936cb4605d6e199772aedc9806c02` | `data/external/katago/katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip` | MIT 式（KataGo） |
| KataGo 主網路 | https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz | 97,898,094 | `9d7a6afed8ff5b74894727e156f04f0cd36060a24824892008fbb6e0cba51f1d` | `data/external/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz` | KataGo Neural Network License（MIT 式） |
| KataGo HumanSL 網路 | https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.bin.gz | 99,066,230 | `637746e44f0efe00ad1245a50aa9bbf0716efe364c43965ead97bd6835d84ab5` | `data/external/katago/b18c384nbt-humanv0.bin.gz` | KataGo Neural Network License（MIT 式） |
| HumanSL PyTorch 檢查點 | https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.ckpt | 323,752,318 | `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27` | `data/external/katago/b18c384nbt-humanv0.ckpt` | KataGo Neural Network License（MIT 式） |
| Strength Estimator 權重（中研院） | https://rlg.iis.sinica.edu.tw/papers/strength-estimator/assets/models/go.tar.gz | 448,813,641 | `885bea594a62956958de526f6a2073f48ecd181535b4d9e72d2c71a339172caf` | `data/external/se/download/go.tar.gz` | 程式庫 GPL-3.0；權重未見授權明示，我們只從權利人公開頁面下載使用，不再散布 |

`go.tar.gz` 內只用 `se_go.pt`（解出後放 `data/external/se/extracted/se_go.pt`，sha256 `95fc295137b1fd09a0ec4d5cc79ed888f8c12d0ba6a77fea58b24efc3f418126`）。`colab/colab_utils.py` 的 `fetch_external()` 會下載並驗證以上全部。

## 已不存在、無法提供的中間檔（要重算）

| 檔 | 用途 | 影響 |
|---|---|---|
| `metric_se.npz`、`metric_hsl.npz`（原在 `.scratch/player_sim3/`） | 舊題 v6 融合的 SE／HSL 度量（`player_sim3.py prep` 產生） | v6 最後一步無法重跑；`metric_hsl` 可由訓練端 HSL 表重建，`metric_se` 需要訓練端 SE 表 |
| `se_uid.parquet`（`.scratch/e21/`） | 訓練端 SE 表（685,231 盤，GPU 12.9 小時） | `metric_se.npz` 重建需要 |
| 舊題 90,000 盤的 SE 特徵（`.scratch/e27/se/se_test_games.parquet`） | v6 融合的 SE 輸入（學校 GPU 約 1 小時） | v6 最後一步無法重跑；可用 `predict_se.py extract` 重抽 |
| pe86 逐局面編碼器權重（`enc_*_{h0,h1,full}.pt`） | 產生每盤 128 維 pe86 向量 | 只剩它產出的向量（中間檔 `pe_test_*_pe86.npy`）；要重算需重訓編碼器（每個 12.6 分鐘，但要先有訓練端全局面抽取約 9.2 小時） |
| 大網路 S（E99）`finh_h0`、`finh_h1` 權重 | booster 訓練池的交叉擬合向量 | 只剩 `finh_full`（Release）；測試端只用 full，booster 已訓練好 |
| `X120_full.npy`、`G_full.npy`、`tok_full.npy`、各種 `pos_*_{trA,trB,val}.npz` | 訓練端 HSL W120 表、主幹向量、逐局面向量 | 要重訓任何網路都得先重新編碼（整條重跑至少約 76 小時 GPU，分項相加） |
| `humanv0.ckpt`、KataGo 網路與執行檔 | 外部資源 | 重新下載（第 3 節，附 sha256） |
