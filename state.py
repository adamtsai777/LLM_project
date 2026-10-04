from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class Workspace:
    dataset: Any = None
    dataset_id: str | None = None
    dataset_name: str = ""
    result: Any = None
    selection: dict = field(default_factory=dict)
    lock: Any = field(default_factory=RLock, repr=False)


def init_state(state):
    if "workspace" not in state:
        state["workspace"] = Workspace()
    if "agent_messages" not in state:
        state["agent_messages"] = []


def set_dataset(state, frame, fingerprint, name):
    workspace = state["workspace"]
    if workspace.dataset_id == fingerprint:
        return
    with workspace.lock:
        workspace.dataset = frame
        workspace.dataset_id = fingerprint
        workspace.dataset_name = name
        workspace.result = None
        workspace.selection = {}
    state["agent_messages"] = []
