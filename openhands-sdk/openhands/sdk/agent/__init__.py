from openhands.sdk.agent.agent import Agent
from openhands.sdk.agent.base import AgentBase


def __getattr__(name: str):
    if name == "ACPAgent":
        from openhands.sdk.agent.acp_agent import ACPAgent

        return ACPAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Agent",
    "AgentBase",
    "ACPAgent",
]
