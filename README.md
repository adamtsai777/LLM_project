# ML 分析與 LangChain Agent

Streamlit 應用，支援 Random Forest、LightGBM、XGBoost 的二元分類、手動超參數調整、模型評估與 SHAP 解釋。LangChain Agent 可呼叫相同 ML 服務，使用 Mistral 生成繁體中文回答。

## 啟動

使用 Python 3.12，在專案目錄執行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

ML 不需要 API 金鑰。Agent 使用環境變數或 `.env` 中的 `MISTRAL_API_KEY`，亦支援 `.streamlit/secrets.toml`，後者優先。可用 `MISTRAL_MODEL` 指定模型，預設 `mistral-small-latest`。請勿將真實金鑰提交到 Git；`.gitignore` 不會自動取消既有檔案的追蹤。

## 使用

### ATP 連線

在 `.env` 加入 `ATP_API_KEY`，重新啟動應用，Agent 分頁會優先選擇 ATP。
按「查詢／更新 ATP 模型」，由官方 `https://api.atptoken.ai/v1/models` 取得可用模型後選擇。
預設優先選取 `claude-haiku-4-5`（必須存在於回傳清單）；可用 `ATP_MODEL` 環境變數指定偏好。
清單也包含非聊天模型，請選擇支援工具呼叫的聊天模型。
使用 `langchain-openai` 的 Chat Completions 介面連線 ATP 官方端點。
仍可切換回 Mistral。切換供應商或模型會清除對話，但保留 ML 成果。
ATP 同樣支援 Streamlit secrets，secrets 中同名金鑰優先於環境變數。
選擇 ATP 後，提示、附件與工具結果會送往 ATP 及所選模型服務。

手動執行真實 API 工具循環驗證（會產生少量用量）：`python -m tests.smoke_atp`。

### 分析操作

1. 在機器學習分頁上傳 CSV；目標須包含 `0/1` 或 `yes/no`，不可缺值。
2. 選擇目標、排除欄位、類別特徵、演算法及預設／手動超參數模式。
3. 訓練後查看 ROC-AUC、Accuracy、Precision、Recall、F1 與混淆矩陣。指標中的正類別為 1。
4. 按「計算 SHAP」，選擇樣本，下載重要性／單筆 SHAP CSV、Waterfall PNG 或評估 JSON。
5. 到 Agent 分頁要求解讀目前模型，或要求「使用 LightGBM，目標為 target，n_estimators 設為 300，訓練後解釋重要特徵」。

Agent 的文字附件只作為提示內容，訓練資料由 ML 分頁上傳。附件上限 40 KB。對話保留於目前 Streamlit session，後續請求會帶入先前訊息和工具結果。LLM 會接收提示、附件及工具輸出的欄位／分析摘要（SHAP 工具包含選定樣本的前十項特徵值）。

## 架構

```text
app.py                  Streamlit 入口
config.py               設定及 Agent 金鑰讀取
state.py                每個 session 獨立的資料、模型與對話狀態
ui/ml_page.py           ML 介面、超參數、結果下載
ui/agent_page.py        對話及附件介面
ml/preprocessing.py    二元目標驗證、補值、One-Hot Encoding
ml/models.py           三種模型、超參數預設及範圍驗證
ml/training.py         分層切分、Pipeline 訓練
ml/evaluation.py       分類評估
ml/explainability.py   SHAP、記憶體內 PNG 產生
agent/builder.py       LangChain create_agent + ChatMistralAI
agent/tools.py         資料檢查、模型設定查詢、訓練、評估、SHAP 工具
agent/prompts.py       角色及工具使用提示
tests/                 ML、Agent 工具循環與 Streamlit 回歸測試
```

前處理只在訓練集 fit；三模型使用相同亂數種子 42 與分層切分。LightGBM 啟用 `subsample_freq=1`，讓抽樣比例實際生效。Random Forest／LightGBM 介面的 `max_depth=0` 代表不限制深度。

Agent 採用 LangChain 官方 [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents) 工具循環。工具綁定該 session 的工作區，模型訓練成功後才替換舊結果。ML 分頁目前選項會提供給 Agent；工具省略欄位與切分設定時沿用 UI，省略模型超參數時則使用該模型預設。

## 範圍與限制

- 目前支援二元分類及手動調參，沒有自動調參或三模型排行榜。
- 只保存最近一次模型，資料與對話不寫入資料庫；重新整理／斷線導致 session 結束後不保證保留。
- 更換資料集會清除模型與對話，修改超參數則保留已完成結果，直到重新訓練。
- SHAP 最多固定抽樣 200 筆測試資料，並限制 dense 展開大小。重要性以編碼後欄位表示。
- Random Forest SHAP 尺度為正類別機率；LightGBM／XGBoost 為 log-odds，不可直接比較數值大小。
- 頻繁利用測試分數調參可能造成對測試集過度擬合；後續自動調參應在訓練集內另做交叉驗證。

## 驗證

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

測試以合成資料實際訓練三種模型、檢查 SHAP 加總與預測一致，並使用可預期的替代 LLM 驗證真正的 LangChain 工具循環，不會呼叫付費 API。
