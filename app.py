import os
import streamlit as st
from mistralai.client import Mistral
from dotenv import load_dotenv
# 載入環境變數

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

UseModel = "mistral-small-latest"
client = Mistral(api_key=api_key)

# 頁面設定，只能設定一次
st.set_page_config(page_title="Morgan Agent Demo ", page_icon="🤖", layout="wide")

st.title("🤖 Morgan Agent Demo")
st.caption("輸入提示詞，按下送出後呼叫 LLM，並保留歷史紀錄。")

# 初始化 session state
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""

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

        submitted = st.form_submit_button("🚀 送出")

    if submitted:
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
                        {"role": "system", "content": instructions_map[role]},
                        {"role": "user", "content": prompt}
                    ])

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