import os
import streamlit as st
from mistralai.client import Mistral
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import re
import matplotlib.pyplot as plt
import shap


# 載入環境變數
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
UseModel = "mistral-small-latest"
client = Mistral(api_key=api_key)

# 頁面設定，只能設定一次
st.set_page_config(page_title="AI Agent Demo ", page_icon="🤖", layout="wide")

# 初始化 session_state
# 一定要放在使用 prompt_history 前面
# =========================
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""

# =========================
# Tabs CSS 樣式
# =========================
st.markdown("""
<style>

/* Tab 整體按鈕 */
button[data-baseweb="tab"] {
    font-size: 48px;
    font-weight: 700;

    background: linear-gradient(to right, #263238, #37474F);

    color: white;

    border-radius: 10px;

    margin-right: 10px;

    padding: 10px 25px;

    height: 60px;
}

/* 被選取的 Tab */
button[data-baseweb="tab"][aria-selected="true"] {

    background: linear-gradient(to right, #00C853, #64DD17);

    color: black;

}

/* Hover 效果 */
button[data-baseweb="tab"]:hover {
    color: #FFD54F;
}

</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 機器學習", "🤖 AI Agent（LangChain）"])


# =========================
# 分頁一：機器學習
# =========================
with tab1:
    
    uploaded_file = st.file_uploader("上傳資料集 CSV", type=["csv"])

    if uploaded_file is not None:

            # =========================
            # 讀取資料
            # =========================
            df = pd.read_csv(uploaded_file)

            st.subheader("📄 資料預覽")
            st.dataframe(df.head())

            st.subheader("📌 欄位資訊")
            st.write(df.columns.tolist())

            # =========================
            # 選擇 Category 欄位
            # =========================
            cat_cols = st.multiselect(
                "選擇 Category 欄位",
                df.columns.tolist()
            )

            # =========================
            # Target 欄位
            # =========================
            target_col = st.selectbox(
                "選擇 Target 欄位",
                df.columns
            )

            # =========================
            # 模型選擇
            # =========================
            model_name = st.selectbox(
                "選擇模型",
                [
                    "RandomForest",
                    "XGBoost"
                ]
            )

            # =========================
            # 開始訓練
            # =========================
            if st.button("🚀 開始模型訓練"):

                try:
                    # =========================
                    # 特徵 / Target
                    # =========================
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    # 轉 category
                    for col in cat_cols:
                        df[col] = df[col].astype("category")

                    # # One-Hot Encoding
                    # X = pd.get_dummies(
                    #     X,
                    #     columns=cat_cols,
                    #     drop_first=True
                    # )
                    
                    # =========================
                    # Target 轉換：yes/no → 1/0
                    # =========================
                    if y.dtype == "object":
                        y = y.astype(str).str.strip().str.lower()

                        y = y.map({
                            "no": 0,
                            "yes": 1,
                            "0": 0,
                            "1": 1
                        })

                    if y.isnull().any():
                        st.error("Target 欄位轉換失敗，請確認 Target 欄位是否只有 yes / no")
                        st.stop()

                    # # 去除空白
                    # y = y.str.strip()

                    # # 小寫化
                    # y = y.str.lower()

                    # 轉換
                    y = y.replace({
                        "yes": 1,
                        "no": 0
                    })

                    # 強制轉數字
                    y = pd.to_numeric(y, errors="coerce")

                    # 檢查失敗資料
                    if y.isnull().any():

                        st.error("Target 欄位包含無法辨識的值")

                        st.write(df[target_col].unique())

                        st.stop()
                    y = y.astype(int)


                    # ---------------------------
                    # 區分數值欄位 / 類別欄位
                    # ---------------------------
                    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
                    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
                    
                    # ---------------------------
                    # 前處理
                    # ---------------------------
                    numeric_transformer = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="median"))
                    ])

                    categorical_transformer = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ])

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", numeric_transformer, numeric_features),
                            ("cat", categorical_transformer, categorical_features)
                        ]
                    )

                    # =========================
                    # 切分資料
                    # =========================
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42,
                        stratify=y
                    )

                    # =========================
                    # 建立模型
                    # =========================
                    if model_name == "RandomForest":

                        model = RandomForestClassifier(
                            n_estimators=100,
                            random_state=42
                        )

                    elif model_name == "XGBoost":

                        model = XGBClassifier(
                            n_estimators=200,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            eval_metric="logloss",
                            random_state=42
                        )

                    clf = Pipeline(steps=[
                        ("preprocessor", preprocessor),
                        ("model", model)
                    ])
                    # =========================
                    # 訓練模型
                    # =========================
                    with st.spinner("模型訓練中..."):

                        # model.fit(X_train, y_train)
                        clf.fit(X_train, y_train)

                    # =========================
                    # 預測
                    # =========================
                    y_pred = clf.predict(X_test)
                    y_proba = clf.predict_proba(X_test)[:, 1]
                    
                    # =========================
                    # 評估
                    # =========================
                    # acc = accuracy_score(y_test, y_pred)

                    st.success("模型訓練完成")

                    st.subheader("📊 模型評估")

                    # print(classification_report(y_test, y_pred, digits=4))
                    # print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
                    
                    st.write(
                        f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}"
                    )
                    
                    st.text(
                        classification_report(
                            y_test,
                            y_pred,
                            digits=4
                        )
                    )


                    

                    # =========================
                    # 混淆矩陣
                    # =========================
                    # cm = confusion_matrix(y_test, y_pred)

                    # fig, ax = plt.subplots()

                    # ax.imshow(cm)

                    # for i in range(cm.shape[0]):
                    #     for j in range(cm.shape[1]):
                    #         ax.text(
                    #             j,
                    #             i,
                    #             cm[i, j],
                    #             ha="center",
                    #             va="center"
                    #         )

                    # ax.set_title("Confusion Matrix")

                    # st.pyplot(fig)

                    # =========================
                    # SHAP 解釋
                    # =========================

                    st.subheader("🔍 SHAP 特徵重要度")

                    # 取得模型
                    model_obj = clf.named_steps["model"]    
                 
                    # 先把測試集經過前處理轉成模型可用格式
                    # X_train_transformed = clf.named_steps["preprocessor"].transform(X_train)
                    X_test_transformed = clf.named_steps["preprocessor"].transform(X_test)
                
                    # 取得前處理後欄位名稱
                    feature_names = clf.named_steps["preprocessor"].get_feature_names_out()

                    # 清洗欄位名稱：移除 XGBoost 不接受的字元
                    clean_feature_names = [
                        re.sub(r'[\[\]<]', '', str(col)).replace(' ', '_')
                        for col in feature_names
]
                    
                    # 建立乾淨欄名的 DataFrame
                    X_test_transformed_df = pd.DataFrame(
                        X_test_transformed,
                        columns=clean_feature_names,
                        index=X_test.index
                    )
                   
                    # =========================
                    # 建立 SHAP Explainer
                    # =========================
                    if model_name == "XGBoost":

                        explainer = shap.TreeExplainer(model_obj)

                    elif model_name == "RandomForest":

                        explainer = shap.TreeExplainer(model_obj)

                    # 計算 SHAP
                    shap_values = explainer.shap_values(X_test_transformed_df)

                    # =========================
                    # 修正重點：取出 class 1 的 SHAP
                    # =========================
                    if isinstance(shap_values, list):
                        shap_values_for_plot = shap_values[1]
                    elif len(shap_values.shape) == 3:
                        shap_values_for_plot = shap_values[:, :, 1]
                    else:
                        shap_values_for_plot = shap_values
                    
                    # plt.figure()
                    # shap.summary_plot(
                    #     shap_values_for_plot,
                    #     X_test_transformed_df,
                    #     show=False
                    # )
                    # plt.tight_layout()
                    # plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
                    # st.pyplot(plt.gcf())
                    # plt.close()
                    
                    # ---------------------------
                    # SHAP Bar Plot
                    # ---------------------------
                    # plt.figure()
                    # shap.summary_plot(
                    #     shap_values_for_plot,
                    #     X_test_transformed_df,
                    #     plot_type="bar",
                    #     show=False
                    # )
                    # plt.tight_layout()
                    # plt.savefig("shap_bar.png", dpi=300, bbox_inches="tight")
                    # st.pyplot(plt.gcf())
                    # plt.close()

                    # ---------------------------
                    # 單一特徵依賴圖（Dependence Plot）
                    # 可改成你想看的特徵名稱
                    # ---------------------------
                    
                    # SHAP 平均重要度
                    shap_importance = np.abs(shap_values_for_plot).mean(axis=0)

                    importance_df = pd.DataFrame({
                        "feature": X_test_transformed_df.columns,
                        "importance": shap_importance
                    })

                    importance_df = importance_df.sort_values(
                        by="importance",
                        ascending=False
                    )

                    candidate_features = importance_df.head(10)["feature"].tolist()

                    st.dataframe(importance_df.head(10))
                    
                    # ---------------------------
                    # 15. 單筆資料解釋（Waterfall Plot）
                    # ---------------------------
                    st.subheader("🔍 單筆資料解釋（Waterfall Plot）")
                    
                    # expected_value 也要取 class 1
                    sample_idx = 0
                    if isinstance(explainer.expected_value, list):
                        base_value = explainer.expected_value[1]
                    elif isinstance(explainer.expected_value, np.ndarray):
                        if explainer.expected_value.ndim > 0:
                            base_value = explainer.expected_value[1]
                        else:
                            base_value = explainer.expected_value
                    else:
                        base_value = explainer.expected_value

                    # 取得單筆 SHAP explanation
                    sample_explanation = shap.Explanation(
                        values=shap_values_for_plot[sample_idx],
                        base_values=base_value,
                        data=X_test_transformed_df.iloc[sample_idx],
                        feature_names=X_test_transformed_df.columns.tolist()
                    )

                    plt.figure()
                    shap.plots.waterfall(
                        sample_explanation,
                        max_display=15,
                        show=False
                    )

                    plt.tight_layout()
                    plt.savefig(
                        "shap_waterfall_sample0.png",
                        dpi=300,
                        bbox_inches="tight"
                    )

                    st.pyplot(plt.gcf())
                    plt.close()
                    
                    
                    
                    # =========================
                    # 單筆 SHAP 數值
                    # =========================
                    st.subheader("🔍 單筆 SHAP 數值")
                    sample_idx = 0

                    # 單筆 SHAP
                    shap_values_sample = shap_values_for_plot[sample_idx]

                    # 建立 DataFrame
                    shap_df = pd.DataFrame({
                        "feature": X_test_transformed_df.columns,
                        "feature_value": X_test_transformed_df.iloc[sample_idx].values,
                        "shap_value": shap_values_sample
                    })

                    # 正負影響
                    shap_df["impact"] = shap_df["shap_value"].apply(
                        lambda x: "positive" if x > 0 else "negative"
                    )

                    # 絕對值排序
                    shap_df["abs_shap"] = shap_df["shap_value"].abs()

                    shap_df = shap_df.sort_values(
                        by="abs_shap",
                        ascending=False
                    )

                    # 顯示前15
                    st.subheader("📊 SHAP 數值 Top10")

                    st.dataframe(
                        shap_df.head(10)
                    )

                    # 匯出 CSV
                    shap_df.to_csv(
                        "shap_values_sample0.csv",
                        index=False,
                        encoding="utf-8-sig"
                    )

                    st.success("SHAP 數值已輸出：shap_values_sample0.csv")    
                            
                except Exception as e:

                    st.error(f"執行失敗：{e}")
 

# =========================
# 分頁二：AI Agent
# =========================
with tab2:
    st.header("🤖 AI Agent")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("提示詞輸入區")

        with st.form("prompt_form"):
            role = st.selectbox(
                "選擇角色",
                ["一般助理", "論文助理", "程式助理", "製造業分析助理"]
            )

            prompt = st.text_area(
                "請輸入提示詞",
                height=250,
                placeholder="請輸入你想要 AI 執行的內容..."
            )
            
            uploaded_file = st.file_uploader(
                "上傳檔案",
                type=[
                    "txt",
                    "csv",
                    "py",
                    "md",
                    "log"
                ]
            )
            
            submitted = st.form_submit_button("🚀 送出")

        if submitted:
            file_content = ""

            if uploaded_file is not None:

                try:

                    # 讀取文字檔
                    file_content = uploaded_file.read().decode("utf-8")

                    st.subheader("📄 檔案內容預覽")

                    st.text(file_content[:3000])

                except Exception as e:

                    st.error(f"檔案讀取失敗：{e}")
            
            full_prompt = prompt

            if file_content:

                full_prompt += f"""

            以下是使用者上傳的檔案內容：

            {file_content}
            """
            
            
            if not api_key:
                st.error("找不到 API_KEY，請先確認 .env 檔案")
            elif prompt.strip():
                try:
                    with st.spinner("AI 思考中..."):
                        instructions_map = {
                            "一般助理": "你是一位專業 AI 助理，請使用繁體中文清楚回答。",
                            "論文助理": "你是一位論文寫作助理，請使用正式、條理清楚的繁體中文回答。",
                            "程式助理": "你是一位資深程式開發助理，請提供清楚、實用的技術建議與範例，並使用繁體中文。",
                            "製造業分析助理": "你是一位製造業與智慧製造分析助理，請從製程、品質、管理與資料分析角度回答，並使用繁體中文。"
                        }

                        response = client.chat.complete(
                            model=UseModel,
                            messages=[
                                {
                                    "role": "system",
                                    "content": instructions_map[role]
                                },
                                {
                                    "role": "user",
                                    "content": full_prompt
                                }
                            ],
                            temperature=0.3
                        )

                        answer = response.choices[0].message.content

                        st.session_state.prompt_history.append({
                            "role": role,
                            "prompt": prompt,
                            "answer": answer
                        })

                        st.session_state.latest_answer = answer
                        st.success("提示詞已送出，AI 回應完成。")

                except Exception as e:
                    st.error(f"執行失敗：{e}")
            else:
                st.warning("請先輸入提示詞")

    with col2:
        st.subheader("歷史提示詞")

        if st.button("清除歷史紀錄"):
            st.session_state.prompt_history = []
            st.session_state.latest_answer = ""
            st.rerun()

        if st.session_state.prompt_history:
            for i, item in enumerate(reversed(st.session_state.prompt_history), 1):
                with st.expander(f"Prompt {i}｜{item['role']}"):
                    st.write("**提示詞：**")
                    st.write(item["prompt"])
                    st.write("**回應：**")
                    st.write(item["answer"])
        else:
            st.info("目前尚無歷史提示詞")

    if st.session_state.prompt_history:
        latest = st.session_state.prompt_history[-1]

        st.divider()
        st.subheader("最新提示詞內容")
        st.write(f"**角色：** {latest['role']}")
        st.code(latest["prompt"], language="text")

        st.subheader("最新 AI 回應")
        st.write(latest["answer"])

# # 初始化 session state
# if "prompt_history" not in st.session_state:
#     st.session_state.prompt_history = []

# if "latest_answer" not in st.session_state:
#     st.session_state.latest_answer = ""

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.subheader("提示詞輸入區")

#     with st.form("prompt_form"):
#         role = st.selectbox(
#             "選擇角色",
#             ["一般助理", "論文助理", "程式助理", "製造業分析助理"]
#         )

#         prompt = st.text_area(
#             "請輸入提示詞",
#             height=250,
#             placeholder="請輸入你想要 AI 執行的內容..."
#         )

#         submitted = st.form_submit_button("🚀 送出")

#     if submitted:
#         if not api_key:
#             st.error("找不到 API_KEY，請先確認 .env 檔案")
#         elif prompt.strip():
#             try:
#                 with st.spinner("AI 思考中..."):
#                     instructions_map = {
#                         "一般助理": "你是一位專業 AI 助理，請使用繁體中文清楚回答。",
#                         "論文助理": "你是一位論文寫作助理，請使用正式、條理清楚的繁體中文回答。",
#                         "程式助理": "你是一位資深程式開發助理，請提供清楚、實用的技術建議與範例，並使用繁體中文。",
#                         "製造業分析助理": "你是一位製造業與智慧製造分析助理，請從製程、品質、管理與資料分析角度回答，並使用繁體中文。"
#                     }

#                     response = client.chat.complete(
#                     model=UseModel,
#                     messages=[
#                         {"role": "system", "content": instructions_map[role]},
#                         {"role": "user", "content": prompt}
#                     ])

#                     answer = response.choices[0].message.content

#                     st.session_state.prompt_history.append({
#                         "role": role,
#                         "prompt": prompt,
#                         "answer": answer
#                     })

#                     st.session_state.latest_answer = answer
#                     st.success("提示詞已送出，AI 回應完成。")

#             except Exception as e:
#                 st.error(f"執行失敗：{e}")
#         else:
#             st.warning("請先輸入提示詞")

# with col2:
#     st.subheader("歷史提示詞")

#     if st.session_state.prompt_history:
#         for i, item in enumerate(reversed(st.session_state.prompt_history), 1):
#             with st.expander(f"Prompt {i}｜{item['role']}"):
#                 st.write("**提示詞：**")
#                 st.write(item["prompt"])
#                 st.write("**回應：**")
#                 st.write(item["answer"])
#     else:
#         st.info("目前尚無歷史提示詞")

# if st.session_state.prompt_history:
#     latest = st.session_state.prompt_history[-1]

#     st.divider()
#     st.subheader("最新提示詞內容")
#     st.write(f"**角色：** {latest['role']}")
#     st.code(latest["prompt"], language="text")

#     st.subheader("最新 AI 回應")
#     st.write(latest["answer"])