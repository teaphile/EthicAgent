"""Agent modules for the EthicAgent pipeline.

Each agent handles one stage of the 8-step pipeline:
  ContextAgent       → Stage 1 (context extraction)
  NeuralReasoner     → Stage 3 (LLM reasoning)
  SymbolicReasoner   → Stage 4 (rule checking)
  FusionAgent        → Stage 5 (neuro-symbolic merge)
  EthicalReasonerAgent → Stage 6 (EDS scoring)
  ActionExecutor     → Stage 7 (decision gate)
  HumanGateway       → (escalation interface)
  ReflectionAgent    → Stage 8 (learning loop)
"""

from ethicagent.agents.context_agent import ContextAgent
from ethicagent.agents.neural_reasoner import NeuralReasoner
from ethicagent.agents.symbolic_reasoner import SymbolicReasoner
from ethicagent.agents.fusion_agent import FusionAgent
from ethicagent.agents.ethical_reasoner import (
    EthicalReasonerAgent,
    EthicalDecision,
    EthicalVerdict,
    PhilosophyResult,
)
from ethicagent.agents.action_executor import ActionExecutor
from ethicagent.agents.human_gateway import HumanGateway
from ethicagent.agents.reflection_agent import ReflectionAgent

__all__ = [
    "ContextAgent",
    "NeuralReasoner",
    "SymbolicReasoner",
    "FusionAgent",
    "EthicalReasonerAgent",
    "EthicalDecision",
    "EthicalVerdict",
    "PhilosophyResult",
    "ActionExecutor",
    "HumanGateway",
    "ReflectionAgent",
]
