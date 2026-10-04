import httpx
from config import ATP_BASE_URL, MISTRAL_MODEL


def list_atp_models(api_key):
    if not api_key:
        raise ValueError("請設定 ATP_API_KEY。")
    try:
        response = httpx.get(
            f"{ATP_BASE_URL}/models", headers={"Authorization": f"Bearer {api_key}"},
            timeout=30, follow_redirects=False,
        )
    except httpx.RequestError:
        raise ValueError("無法連線 ATP，請檢查網路後重試。") from None
    if response.status_code != 200:
        raise ValueError(f"ATP 模型查詢失敗（HTTP {response.status_code}），請檢查金鑰與專案權限。")
    try:
        ids = sorted({item["id"] for item in response.json()["data"] if isinstance(item.get("id"), str)})
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("ATP 模型清單格式不正確。") from None
    if not ids:
        raise ValueError("ATP 專案尚未提供可用模型。")
    return ids


def create_chat_model(provider, api_key, model_name=None):
    if not api_key:
        raise ValueError(f"請設定 {provider.upper()}_API_KEY。")
    if provider == "ATP":
        from langchain_openai import ChatOpenAI
        if not model_name:
            raise ValueError("請先查詢 ATP 模型並選擇模型。")
        return ChatOpenAI(
            api_key=api_key, base_url=ATP_BASE_URL, model=model_name,
            use_responses_api=False, stream_usage=False,
            timeout=60, max_retries=1,
        )
    if provider == "Mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=model_name or MISTRAL_MODEL, mistral_api_key=api_key,
            temperature=0.3, timeout=60, max_retries=1,
        )
    raise ValueError("不支援的 LLM 供應商。")
