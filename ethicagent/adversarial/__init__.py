"""Adversarial testing module for EthicAgent.

Provides three attack surfaces:
  - **Perturbation**: subtle input mutations (typos, synonyms, paraphrasing)
  - **Jailbreak**: prompt-injection / system-prompt override attempts
  - **Robustness**: aggregate robustness evaluation combining both

Usage::

    from ethicagent.adversarial import PerturbationAttack, JailbreakAttack, RobustnessEvaluator

    perturb = PerturbationAttack()
    results = perturb.run(cases)

    jailbreak = JailbreakAttack()
    results = jailbreak.run(cases)

    evaluator = RobustnessEvaluator()
    report = evaluator.run(cases)
"""

from ethicagent.adversarial.perturbation import PerturbationAttack
from ethicagent.adversarial.jailbreak import JailbreakAttack
from ethicagent.adversarial.robustness import RobustnessEvaluator

__all__ = [
    "PerturbationAttack",
    "JailbreakAttack",
    "RobustnessEvaluator",
]
