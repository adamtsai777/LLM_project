import streamlit as st
from hashlib import sha256
from config import MAX_ATTACHMENT_BYTES, get_api_key, ATP_MODEL, MISTRAL_MODEL
from agent.providers import list_atp_models
from agent.prompts import ROLES


def render_agent_page():
    st.subheader("LangChain AI Agent")
    st.caption("可查詢目前資料、訓練模型、解讀評估與 SHAP，並接續對話追問。")
    provider = st.selectbox("LLM 供應商", ["ATP", "Mistral"],
                            index=0 if get_api_key("ATP") else 1)
    api_key = get_api_key(provider)
    model_name = MISTRAL_MODEL if provider == "Mistral" else None
    if provider == "ATP" and api_key:
        fingerprint = sha256(api_key.encode()).hexdigest()
        if st.session_state.get("atp_key_fingerprint") != fingerprint:
            st.session_state.atp_key_fingerprint = fingerprint
            st.session_state.atp_models = []
        if st.button("查詢／更新 ATP 模型"):
            try:
                with st.spinner("查詢 ATP 可用模型…"):
                    st.session_state.atp_models = list_atp_models(api_key)
            except ValueError as error:
                st.session_state.atp_models = []
                st.error(str(error))
        models = st.session_state.get("atp_models", [])
        if models:
            model_name = st.selectbox("ATP 模型", models,
                                      index=models.index(ATP_MODEL) if ATP_MODEL in models else 0)
            st.caption("請選擇支援工具呼叫的聊天模型；清單也可能包含圖片、影片或嵌入模型。")
        else:
            st.info("請先按「查詢／更新 ATP 模型」取得這把金鑰可用的模型。")
    context = (provider, model_name)
    if st.session_state.get("agent_provider_context", context) != context:
        st.session_state.agent_messages = []
    st.session_state.agent_provider_context = context
    st.caption("切換供應商或模型會清除對話，已訓練的 ML 模型仍保留。")
    if st.button("清除對話紀錄"):
        st.session_state.agent_messages = []
    for message in st.session_state.agent_messages:
        if message.type in ("human", "ai") and message.content:
            with st.chat_message("user" if message.type == "human" else "assistant"):
                st.write(message.content)
        elif message.type == "tool":
            with st.expander(f"工具結果：{message.name}"):
                st.text(message.content)
    if not api_key:
        st.info(f"設定 {provider.upper()}_API_KEY 後即可使用 Agent；機器學習功能可獨立使用。")
    with st.form("agent_form", clear_on_submit=True):
        role = st.selectbox("助理角色", list(ROLES))
        prompt = st.text_area("問題或分析需求", placeholder="使用 LightGBM 訓練，n_estimators 設為 300，並解釋重要特徵。")
        attachment = st.file_uploader("文字附件（選填；訓練資料請至機器學習分頁上傳）",
                                       type=["txt", "csv", "py", "md", "log"], key="agent_attachment")
        submitted = st.form_submit_button("送出", disabled=not api_key or not model_name)
    if not submitted:
        return
    if not prompt.strip():
        st.warning("請先輸入問題。")
        return
    try:
        from agent.builder import build_agent
        from langchain_core.messages import HumanMessage
        if attachment is not None:
            content = attachment.getvalue()
            if len(content) > MAX_ATTACHMENT_BYTES:
                raise ValueError("文字附件上限為 40 KB，請縮小內容後重試。")
            prompt += "\n\n<使用者附件>\n" + content.decode("utf-8-sig") + "\n</使用者附件>"
        agent = build_agent(st.session_state.workspace, api_key, role,
                            provider=provider, model_name=model_name)
        messages = [*st.session_state.agent_messages, HumanMessage(content=prompt)]
        with st.spinner("Agent 正在分析…"):
            response = agent.invoke({"messages": messages}, config={"recursion_limit": 20})
        st.session_state.agent_messages = response["messages"]
        st.rerun()
    except Exception as error:
        detail = str(error).replace(api_key, "[REDACTED]") if api_key else str(error)
        st.error(f"Agent 執行失敗：{detail}")
        st.caption("已完成的工具訓練仍會保留於機器學習分頁；可以重新送出問題。")
