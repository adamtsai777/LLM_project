import streamlit as st

from state import init_state
from ui.ml_page import render_ml_page
from ui.agent_page import render_agent_page

st.set_page_config(page_title="ML 分析與 AI Agent", page_icon="🤖", layout="wide")
init_state(st.session_state)
st.title("ML 分析與 AI Agent")
ml_tab, agent_tab = st.tabs(["📊 機器學習", "🤖 AI Agent"])
with ml_tab:
    render_ml_page()
with agent_tab:
    render_agent_page()
