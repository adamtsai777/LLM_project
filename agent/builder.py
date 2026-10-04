from langchain.agents import create_agent
from agent.providers import create_chat_model
from agent.prompts import ROLES, SYSTEM_PROMPT
from agent.tools import build_tools


def build_agent(workspace, api_key=None, role="一般助理", model=None,
                provider="Mistral", model_name=None):
    if model is None:
        model = create_chat_model(provider, api_key, model_name)
    return create_agent(
        model=model, tools=build_tools(workspace),
        system_prompt=ROLES.get(role, ROLES["一般助理"]) + SYSTEM_PROMPT,
    )
