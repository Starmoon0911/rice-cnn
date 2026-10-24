# 水稻葉片病害 CNN

## 預訓練模型比較（依 models.md）

新增入口 `src/compare_pretrained.py`，與下方既有從零訓練／Grad-CAM 實驗分開。
**訓練請手動執行**；程式依序完成所選模型，每完成一個就更新比較表。

```powershell
# 全部九個模型，預設 Intel XPU、30 epochs、batch size 8
.\.venv\Scripts\python.exe src/compare_pretrained.py

# 或逐個／分批執行；共用相同設定與 run-dir 即可累積結果
.\.venv\Scripts\python.exe src/compare_pretrained.py --models resnet18
.\.venv\Scripts\python.exe src/compare_pretrained.py --models densenet121 efficientnet_b0 mobilenet_v3_small
.\.venv\Scripts\python.exe src/compare_pretrained.py --models inception_v3 se_resnet18 cbam_resnet18 convnext_tiny rice_cnn

# 完成後只彙整，不訓練、不需要 GPU
.\.venv\Scripts\python.exe src/compare_pretrained.py --summarize-only --run-dir results/pretrained_xpu

# CPU 或新的實驗（更改訓練設定時請使用新的 run-dir）
.\.venv\Scripts\python.exe src/compare_pretrained.py --device cpu --models resnet18 --run-dir results/pretrained_cpu
```

模型名稱：`rice_cnn`（Simple CNN）、`resnet18`、`densenet121`、`efficientnet_b0`、
`mobilenet_v3_small`、`inception_v3`、`se_resnet18`、`cbam_resnet18`、`convnext_tiny`。
MobileNetV3 使用 Small 版本，延續專案原本的選擇。

- Simple CNN 為現有 RiceCNN，沒有官方預訓練權重，從隨機初始化訓練。
- 其他模型使用 torchvision ImageNet `DEFAULT` 權重，再替換為資料集分類頭；
  實際 enum 名稱、torch／torchvision 版本會保存，下載失敗會報錯，不會改用隨機權重。
- SE／CBAM 是自訂變體：先載入 ResNet18 預訓練權重，在每個 BasicBlock 第二個 BN 後、
  殘差相加前加入 attention。骨幹保留，新增模組與分類頭隨機初始化，所有參數一起 fine-tune。
- 驗證／測試使用各權重的官方 resize、center crop、interpolation 與 normalization。
  訓練使用 RandomResizedCrop（scale 0.8–1.0）、水平翻轉、15 度旋轉及 color jitter 0.15。
  一般模型為 224×224，InceptionV3 為 299×299，且訓練 loss 為主頭 CE + 0.4 × 輔助頭 CE。
- 共用固定 seed 42、70%／15%／15% 分層切分、AdamW（lr 0.0003、weight decay 0.0001），
  完整跑指定 epochs，以最小 validation CE 選 checkpoint，最後才測試。
  因各模型輸入解析度／權重配方不同，這是預訓練系統比較，不是純架構消融。
- 檔案 SHA-256、完整切分與訓練設定會保存；相同目錄拒絕不同設定。
  預設略過已有 checkpoint 或 metrics 的模型，保留既有檔案並繼續訓練其餘模型；全部略過時仍更新比較表。
  `--overwrite` 可明確重訓所選模型；中斷後也可用此選項從頭重訓，沒有續接 optimizer 功能。
  若 XPU 記憶體不足，可將**整個新實驗**改為 `--batch-size 4 --run-dir results/pretrained_xpu_b4`。
- 固定 seed 並優先使用 deterministic 運算；不支援的運算會警告，跨硬體不保證逐位元重現。
  訓練秒數含訓練、驗證與 checkpoint／history 寫入，排除權重下載和最終測試，並同步 XPU 計時。
- EfficientNet-B0 的 SE 層在 XPU 上以等價 linear 運算處理池化後的 1×1 卷積，
  避免部分批次大小（本機 batch size 4）觸發卷積 bias 梯度錯誤，造成驗證 loss 為 NaN。
  預訓練權重與 checkpoint 的參數名稱、形狀保持相容；CPU 使用原本卷積。

預設輸出 `results/pretrained_xpu/`（CPU 為 `results/pretrained_cpu/`）：

- `comparison.md`／`.csv`／`.json`：九模型總表，未完成模型維持待測。
- `<model>_metrics.json`：Macro F1、accuracy、各類 precision／recall／F1、混淆矩陣、
  每張測試圖片的預測機率、最佳 epoch、參數量與訓練秒數。
- `<model>_history.json`：每個 epoch 的訓練 loss 與驗證指標。
- `checkpoints/<model>.pth`：CPU 可攜式最佳權重、分類順序、前處理與實驗設定。
- `split.json`、`experiment.json`：重現／比對實驗所需資訊。

依使用者指定，**LAR 與病斑遮蔽後信心下降保持待測**，不以 CAM 熱區代替病斑標註。
`models.md` 原始表格保留；實際測量寫入上述輸出，避免不同實驗互相覆蓋。
訓練完成後提供 `comparison.md` 與各模型 metrics／history，即可分析分類表現、混淆類別、
過擬合與時間取捨。現有資料量小，單一切分與 seed 的差異只適合作探索性比較。

官方參考：[torchvision models](https://docs.pytorch.org/vision/main/models)、
[InceptionV3](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html)。

## Intel XPU 訓練

兩個訓練入口預設使用 Intel GPU（`--device xpu`），XPU 不可用時會直接報錯。可用 `--device cpu` 明確切回 CPU。
依照 [PyTorch 官方 Intel GPU 文件](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)，使用 XPU 版套件及相容的 Intel GPU 驅動：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
.\.venv\Scripts\python.exe test.py
.\.venv\Scripts\python.exe src/train.py --device xpu
.\.venv\Scripts\python.exe src/compare_models.py --device xpu
```

`train.py` 預設訓練 50 epochs，儲存 `models/best_rice_cnn_xpu.pth`、`results/xpu/` 與 `figures/xpu/`；可用 `--epochs`、`--model-dir`、`--result-dir`、`--figure-dir` 調整。
`compare_models.py` 的 XPU 預設輸出為 `models/comparison_xpu/`、`results/comparison_xpu/`、`figures/model_comparison_xpu/`。CPU 模式保留原路徑。重跑會覆寫所選目錄中的同名結果。
模型與批次資料在 XPU 上訓練及驗證；權重以 CPU tensor 儲存，可在沒有 Intel GPU 的電腦載入。多模型比較的測試評估與 Grad-CAM 產圖仍在 CPU 執行。

解釋新訓練的單模型權重：

```powershell
.\.venv\Scripts\python.exe src/grad_cam.py --checkpoint models/best_rice_cnn_xpu.pth --output-dir figures/grad_cam_xpu
```

## 多模型 Grad-CAM 視覺比較

開啟 `figures/model_comparison_xpu/index.html` 可查看 XPU 訓練後的完整圖庫（原 CPU 圖庫在 `figures/model_comparison/index.html`）；`overview.png` 是每類最多三張的總覽，`class_0.png` 到 `class_2.png` 是各病害的全部測試圖片，每張圖片也有單獨的橫向比較圖。

這一輪比較四種架構，**全部從隨機權重開始訓練，未使用 ImageNet 預訓練權重**：

| 模型 | 架構類型 | Grad-CAM 取樣層 | 128×128 輸入的原生熱區 |
|---|---|---|---|
| RiceCNN | 一般 CNN | 最後一層卷積 `features.12` | 16×16 |
| ResNet18 | 殘差 CNN | 最後一個殘差區塊 `layer4.1` | 4×4 |
| MobileNetV3-Small | 深度可分離卷積 CNN | 最後卷積區塊 `features.12` | 4×4 |
| Swin-T | 視窗注意力 Transformer | 最後空間 `norm`，NHWC 轉 NCHW | 4×4 |

執行訓練與產圖：

```powershell
.\.venv\Scripts\python.exe src/compare_models.py
```

預設 XPU 訓練、CPU 4 threads、30 epochs、batch size 12、AdamW（learning rate 0.0003、weight decay 0.0001）、128×128 輸入及固定 seed 42。模型共用相同資料切分與資料增強設定，各自以最低驗證集 cross entropy 選擇 checkpoint，不以測試結果選模型。執行會重寫這個比較實驗的同名權重與結果；原有的 `models/best_rice_cnn_cpu.pth` 不受影響。

檔案路徑排序後，分層切分為 84 張訓練、18 張驗證、18 張測試；**比較圖片只取自本輪測試集**。舊版訓練未保存完整切分，不能確認舊權重使用相同圖片，所以本輪也重新訓練 RiceCNN。`results/comparison_xpu/split.json` 保存每張圖片的路徑、SHA-256 與切分索引，checkpoint 保存對應 split ID。這是圖片層級切分；資料未提供植株／拍攝批次分組，不能保證來自不同植株。

預設所有模型都解釋圖片的**真實類別**，因此即使預測不同，熱區仍對應相同目標；預測錯誤以紅字標示。沒有正向證據的熱區會註明 `No positive evidence`。每張熱區各自正規化，不能用顏色比較模型的絕對證據強度。各模型空間解析度不同，較銳利的熱區不代表病斑定位較準；沒有病斑標註時不提供定位排名。

訓練完僅重新產圖：

```powershell
.\.venv\Scripts\python.exe src/compare_models.py --render-only
```

指定所有圖片都解釋 `Brown spot`，另存一份圖庫：

```powershell
.\.venv\Scripts\python.exe src/compare_models.py --render-only --target "Brown spot" --output-dir figures/model_comparison_brown_spot
```

其他選項：`--models rice_cnn resnet18` 選擇模型；`--samples-per-class 3` 限制每類張數（0 為全部測試圖片）；`--epochs`、`--seed`、`--image-size`、`--learning-rate` 可修改訓練設定。重新產圖時 `seed` 和 `image-size` 必須與原實驗一致，也會核對各模型的訓練設定，拒絕混用不同訓練預算的舊結果。若要保留另一組實驗，請同時另設 `--model-dir`、`--result-dir`、`--output-dir`。

輸出內容：

- `models/comparison_xpu/*.pth`：每種架構的最佳驗證權重。
- `results/comparison_xpu/`：資料切分、實驗設定、每輪訓練紀錄及測試分類指標。
- `figures/model_comparison_xpu/summary.json`：圖片來源、各模型預測、共同解釋類別、空白熱區與原生解析度。
- `figures/model_comparison_xpu/heatmaps.npz`：未套色的 128×128 熱區；key 為模型名稱加圖庫的三位數序號，例如 `resnet18_001`。

此結果只是一個小資料集、單一 seed、固定訓練預算的探索性比較。分類準確率與病斑關注位置分開判讀；尤其 Swin-T 等架構從零訓練的結果，不能代表預訓練或充分調參後的表現。

## Grad-CAM：檢查模型關注的位置

使用現有的 `models/best_rice_cnn_cpu.pth`，不需重新訓練或安裝額外套件。在專案根目錄執行：

```powershell
.\.venv\Scripts\python.exe src/grad_cam.py
```

預設以固定亂數種子從每個類別抽三張圖片，輸出至 `figures/grad_cam/`：

- `overview.png`：所有選取圖片的熱區疊圖；預測錯誤的標題以紅色顯示。
- 個別 PNG：模型輸入、熱區圖、疊圖並排，附圖片檔名、資料夾標籤、預測類別及 softmax 分數。
- `summary.json`：圖片來源、每類預測機率、解釋的目標類別，以及是否為空白熱區。

指定一張圖片：

```powershell
.\.venv\Scripts\python.exe src/grad_cam.py --image "dataset/rice_leaf_diseases/Leaf smut/DSC_0500.jpg"
```

預設解釋「預測類別」。若模型判斷錯誤，可指定真實類別比較其熱區，並使用不同輸出目錄保留結果：

```powershell
.\.venv\Scripts\python.exe src/grad_cam.py --image "dataset/rice_leaf_diseases/Leaf smut/DSC_0500.jpg" --target-class "Leaf smut" --output-dir figures/grad_cam_true_class
```

`--target-class` 可使用 `Bacterial leaf blight`、`Brown spot` 或 `Leaf smut`。檢查全部圖片：

```powershell
.\.venv\Scripts\python.exe src/grad_cam.py --samples-per-class 0 --output-dir figures/grad_cam_all
```

也可用 `--checkpoint`、`--data-dir`、`--output-dir` 指定其他路徑。相同輸出目錄中的同名結果會被覆寫。

### 如何判讀

先比較原圖與疊圖，確認高熱區是否落在葉片、病斑或枯黃區域；若集中在背景、邊框或其他物件，表示模型可能依賴不適當的線索。尤其要查看紅色標題的錯誤預測，並與真實類別的熱區比較。

圖中輸入與訓練的評估流程一致：RGB、縮放至 checkpoint 指定大小（目前為 128×128），並套用相同的 ImageNet mean/std 正規化。Grad-CAM 使用最後一層卷積 `features[12]` 的目標類別 logit 梯度，對空間梯度取平均作為各通道權重，再加權特徵圖、套用 ReLU 與雙線性放大。模型架構共用於 `src/model.py`，載入既有權重時會嚴格檢查相容性。

熱區的紅色代表該張圖片中較高的正向證據，每張圖片各自正規化至 0–1，不能跨圖比較絕對強度。疊圖的透明度隨熱度改變，以便看清病斑；若沒有正向證據，會顯示提示且不加上彩色覆蓋。原始特徵圖只有 16×16，因此放大後的區域是粗略定位，不能當作病斑邊界。

Grad-CAM 是輔助檢查工具，不能單獨證明模型學到正確病徵。資料集目前沒有病斑位置標註，無法計算病斑重疊率；預設抽樣也可能包含訓練圖片，不代表獨立測試結果。若要量化確認，需要另外標註病斑，並在未參與訓練的圖片上評估。顯示的 softmax 分數也不是經校準的正確率。

方法來源：[Grad-CAM 原始論文（ICCV 2017）](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)。

### 驗證

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

測試涵蓋已知區域的類別專屬熱區、無正向證據時的空白熱區、hook 清理、模型梯度不被累積，以及使用現有權重產生 PNG／JSON 的整合流程。
