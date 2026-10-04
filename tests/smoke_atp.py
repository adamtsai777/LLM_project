"""Manual live smoke test; sends only a synthetic request, never a user dataset."""
from dotenv import dotenv_values
from agent.builder import build_agent
from agent.providers import list_atp_models
from state import Workspace


def main():
    key = dotenv_values(".env").get("ATP_API_KEY")
    try:
        models = list_atp_models(key)
        model = "claude-haiku-4-5"
        assert model in models
        agent = build_agent(Workspace(), api_key=key, provider="ATP", model_name=model)
        result = agent.invoke({"messages": [{"role": "user", "content":
            "請呼叫 get_model_results 工具檢查是否已有模型。不要訓練，最後用一句繁體中文回答。"}]},
            config={"recursion_limit": 8})
        calls = [message.name for message in result["messages"] if message.type == "tool"]
        assert "get_model_results" in calls, "Tool was not called"
        assert result["messages"][-1].content, "Missing final answer"
        print("PASS: ATP authentication, model discovery, LangChain tool call and final answer")
        print("Model:", model)
        print("Tools:", calls)
    except Exception as error:
        print("FAIL:", type(error).__name__, str(error).replace(key or "unused-key", "[REDACTED]"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
