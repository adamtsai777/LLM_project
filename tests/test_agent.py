import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from agent.builder import build_agent
from agent.tools import build_tools
from state import Workspace


class ScriptedToolModel(BaseChatModel):
    """A deterministic LLM substitute that exercises the real LangChain agent loop."""
    @property
    def _llm_type(self):
        return "scripted-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            response = AIMessage(content="", tool_calls=[{
                "name": "train_classifier", "args": {
                    "model_name": "LightGBM", "target": "target", "params": {"n_estimators": 10},
                }, "id": "train-1", "type": "tool_call",
            }])
        elif isinstance(last, ToolMessage) and last.name == "train_classifier":
            response = AIMessage(content="", tool_calls=[{
                "name": "get_model_results", "args": {}, "id": "results-1", "type": "tool_call",
            }])
        else:
            response = AIMessage(content="已依照工具結果完成模型訓練與評估。")
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_real_agent_tool_loop_and_session_isolation():
    workspace = Workspace(dataset=pd.DataFrame({"x": list(range(80)), "target": [0, 1] * 40}))
    agent = build_agent(workspace, model=ScriptedToolModel())
    result = agent.invoke({"messages": [HumanMessage(content="訓練 LightGBM")]})
    assert workspace.result.model_name == "LightGBM"
    assert workspace.result.params["n_estimators"] == 10
    assert sum(isinstance(message, ToolMessage) for message in result["messages"]) == 2
    assert result["messages"][-1].content == "已依照工具結果完成模型訓練與評估。"
    followup = agent.invoke({"messages": [*result["messages"], HumanMessage(content="再訓練一次")]})
    assert len(followup["messages"]) > len(result["messages"])
    other_tools = {tool.name: tool for tool in build_tools(Workspace())}
    assert "error" in other_tools["get_model_results"].invoke({})


def test_tool_failures_preserve_result():
    workspace = Workspace(dataset=pd.DataFrame({"x": list(range(80)), "target": [0, 1] * 40}))
    tools = {tool.name: tool for tool in build_tools(workspace)}
    tools["train_classifier"].invoke({"model_name": "Random Forest", "target": "target", "params": {"n_estimators": 10}})
    old = workspace.result
    failed = tools["train_classifier"].invoke({"model_name": "XGBoost", "target": "missing"})
    assert "error" in failed
    assert workspace.result is old
    explanation = tools["explain_predictions"].invoke({"sample_idx": 0})
    assert explanation["scale"] == "probability"
    assert explanation["run_id"] == old.run_id
