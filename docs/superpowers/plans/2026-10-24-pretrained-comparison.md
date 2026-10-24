# 預訓練模型比較實作計畫

使用者授權：實作 models.md 九種模型，訓練由使用者手動執行，取得結果後分析。
LAR 與病斑遮蔽後信心下降依使用者回覆保持待測。

設計：新增獨立入口，保留現有從零訓練與 Grad-CAM 流程。MobileNetV3
採現有 Small 版本。Simple CNN 隨機初始化；SE/CBAM 使用預訓練 ResNet18
骨幹，在每個 BasicBlock 殘差相加前加入 attention；新 attention 與分類頭隨機初始化。
其餘使用 torchvision DEFAULT，記錄解析後的確切權重名稱與版本。
各模型使用權重指定的驗證前處理，訓練使用相同類型的資料增強及模型對應解析度。
這是各模型預訓練配方的比較，不是固定輸入解析度的純架構消融。

1. 新增 tests/test_pretrained_models.py：九模型 forward、attention 梯度、
   預訓練骨幹保留、Inception 訓練輸出與 299 前處理。
2. 新增 src/pretrained_models.py：模型工廠、SE/CBAM BasicBlock、前處理、
   Inception 主頭與輔助頭 loss。
3. 新增 src/compare_pretrained.py：固定 70/15/15 分層切分及內容雜湊，
   共用 AdamW、seed、epoch、batch size，驗證 loss 選模，逐模型保存
   CPU checkpoint、曲線 JSON、test 分類報告、混淆矩陣、逐張預測、訓練秒數。
   實驗設定不同則拒絕共用目錄，已有結果須明確 --overwrite。
4. 新增結果彙整測試；輸出 comparison.json/.csv/.md，不覆寫 models.md 待測表。
   --summarize-only 不需 GPU 或預訓練下載。
5. README 補齊手動命令、路徑、比較限制；執行完整 unittest 與 CLI help 驗證。

環境：目錄沒有 .git，git 也不在 PATH，直接在使用者指定工作區新增檔案。
不下載權重、不執行真實資料訓練。以無下載 forward/backward 與合成資料測試驗證。

執行紀錄：步驟 1–5 已實作。完整 unittest 18 項通過；九模型在本機 Intel XPU
以合成 batch 2（224／299）通過 train-mode forward/backward，不載入或訓練真實資料。
實際資料 manifest 驗證為 84／18／18。獨立唯讀 review 無重大問題；依 minor 建議
將 checkpoint 改為暫存檔寫入後原子替換，並重新執行完整測試。
未做預訓練檔案實際下載或正式訓練；預訓練骨幹保留以替代下載的真實 state_dict 測試。
