import os
from dotenv import load_dotenv

load_dotenv()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
ATP_BASE_URL = "https://api.atptoken.ai/v1"
ATP_MODEL = os.getenv("ATP_MODEL", "claude-haiku-4-5")
RANDOM_STATE = 42
MAX_SHAP_ROWS = 200
MAX_SHAP_CELLS = 2_000_000
MAX_ATTACHMENT_BYTES = 40_000


def get_api_key(provider="Mistral"):
    """Only the Agent needs credentials; missing secrets must not block ML."""
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError
    name = "ATP_API_KEY" if provider == "ATP" else "MISTRAL_API_KEY"
    try:
        secret = st.secrets.get(name)
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        secret = None
    return secret or os.getenv(name)
