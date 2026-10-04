from hashlib import sha256
from io import BytesIO
import json
import pandas as pd
import streamlit as st
from ml.models import MODEL_PARAMS, resolve_params
from ml.training import train_model
from ml.explainability import explain_model, waterfall_png
from state import set_dataset


def parameter_inputs(model_name):
    manual = st.radio("超參數模式", ["預設參數", "手動調參"], horizontal=True)
    if manual == "預設參數":
        params = resolve_params(model_name)
        with st.expander("查看預設超參數"):
            st.json(params)
        return params
    params = {}
    columns = st.columns(3)
    for idx, (name, spec) in enumerate(MODEL_PARAMS[model_name].items()):
        kind, default, *limits = spec
        key = f"param_{model_name}_{name}"
        label = name
        if name == "max_depth" and model_name != "XGBoost":
            label += "（0 = 不限制）"
        with columns[idx % 3]:
            if kind == "select":
                params[name] = st.selectbox(label, limits[0], key=key,
                                           format_func=lambda x: "全部特徵" if x is None else x)
            else:
                params[name] = st.number_input(
                    label, min_value=limits[0], max_value=limits[1], value=default,
                    step=1 if kind == "int" else 0.001,
                    format="%d" if kind == "int" else "%.3f", key=key,
                )
    return params


def render_results(workspace):
    result = workspace.result
    if result is None:
        return
    st.divider()
    st.subheader(f"最近一次訓練結果：{result.model_name}")
    st.caption(f"目標：{result.target}｜訓練 {result.train_rows} 筆｜測試 {len(result.X_test)} 筆｜正類別：1")
    st.caption("以下呈現已完成的訓練結果；修改上方設定後，需重新訓練才會更新。")
    for col, (name, value) in zip(st.columns(5), result.evaluation["metrics"].items()):
        col.metric(name, f"{value:.4f}")
    with st.expander("實際訓練設定與分類報告"):
        st.json(result.summary())
        st.text(result.evaluation["report"])
    st.write("混淆矩陣（列為實際類別，欄為預測類別）")
    st.dataframe(pd.DataFrame(result.evaluation["confusion_matrix"],
                              index=["實際 0", "實際 1"], columns=["預測 0", "預測 1"]))
    st.download_button("下載評估 JSON", json.dumps(result.summary(), ensure_ascii=False, indent=2),
                       file_name="model_evaluation.json", mime="application/json")
    st.subheader("SHAP 模型解釋")
    st.caption("最多使用 200 筆測試資料計算，類別特徵會以編碼後欄位呈現。")
    if result.explanation is None and st.button("計算 SHAP"):
        try:
            with st.spinner("計算 SHAP 中…"), workspace.lock:
                explain_model(result)
        except Exception as error:
            st.error(f"SHAP 計算失敗（訓練結果仍保留）：{error}")
    explanation = result.explanation
    if explanation is None:
        return
    st.caption(f"解釋 {len(explanation.data)} 筆｜尺度：{explanation.scale}｜正類別：1；不同尺度不可直接比較。")
    st.dataframe(explanation.importance.head(10), hide_index=True)
    st.download_button("下載特徵重要性 CSV", explanation.importance.to_csv(index=False).encode("utf-8-sig"),
                       file_name="shap_importance.csv", mime="text/csv")
    sample_idx = st.selectbox("選擇解釋樣本（測試子集位置）", range(len(explanation.data)),
                              key=f"sample_{result.run_id}")
    st.caption(f"原始資料索引：{explanation.data.index[sample_idx]}")
    try:
        png = waterfall_png(explanation, sample_idx)
        st.image(png)
        st.download_button("下載 Waterfall 圖", png, file_name=f"shap_sample_{sample_idx}.png", mime="image/png")
        table = explanation.sample_table(sample_idx)
        st.dataframe(table.head(10), hide_index=True)
        st.download_button("下載單筆 SHAP CSV", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"shap_sample_{sample_idx}.csv", mime="text/csv")
    except Exception as error:
        st.error(f"無法繪製 SHAP：{error}")


def render_ml_page():
    workspace = st.session_state.workspace
    upload = st.file_uploader("上傳資料集 CSV", type=["csv"], key="dataset_upload")
    if upload is not None:
        content = upload.getvalue()
        fingerprint = sha256(content).hexdigest()
        if fingerprint != workspace.dataset_id:
            try:
                df = pd.read_csv(BytesIO(content))
                if df.empty or len(df.columns) < 2:
                    raise ValueError("CSV 至少需要一筆資料、目標欄位與一個特徵。")
                set_dataset(st.session_state, df, fingerprint, upload.name)
            except Exception as error:
                st.error(f"CSV 讀取失敗：{error}")
                st.info("新檔案未載入；先前資料與結果保留。請重新上傳有效 CSV。")
                return
    if workspace.dataset is None:
        st.info("請上傳 CSV 開始分析；目標欄位須為 0/1 或 yes/no。")
        return
    df = workspace.dataset
    st.caption(f"目前資料：{workspace.dataset_name}｜{len(df)} 筆 × {len(df.columns)} 欄；更換資料會清除舊模型與對話。")
    st.dataframe(df.head(10), hide_index=True)
    prefix = workspace.dataset_id
    target = st.selectbox("目標欄位", df.columns.tolist(), key=f"target_{prefix}")
    features = [col for col in df if col != target]
    drop_cols = st.multiselect("不使用的欄位", features, key=f"drop_{prefix}_{target}")
    cat_cols = st.multiselect("指定為類別的特徵（文字欄位會自動辨識）",
                              [col for col in features if col not in drop_cols],
                              key=f"cat_{prefix}_{target}_{tuple(drop_cols)}")
    model_name = st.selectbox("演算法", list(MODEL_PARAMS))
    params = parameter_inputs(model_name)
    test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
    workspace.selection = {"target": target, "drop_cols": drop_cols,
                           "categorical_cols": cat_cols, "test_size": test_size,
                           "model_name": model_name, "params": params}
    if st.button("開始模型訓練", type="primary"):
        try:
            with st.spinner(f"正在訓練 {model_name}…"), workspace.lock:
                workspace.result = train_model(df, target, model_name, params, drop_cols, cat_cols, test_size)
            st.success("模型訓練完成，可切換至 Agent 解讀結果。")
        except Exception as error:
            st.error(f"訓練失敗：{error}")
    render_results(workspace)
