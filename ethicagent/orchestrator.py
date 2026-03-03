"""EthicAgent Orchestrator — the main pipeline controller.

Runs the full 8-stage ethical reasoning pipeline:

    1. Context Extraction — ContextAgent enriches raw input
    2. Knowledge Retrieval — pull domain rules & constraints from KG
    3. Neural Reasoning  — LLM-based ethical analysis
    4. Symbolic Reasoning — hard / duty / domain rule checking
    5. Neuro-Symbolic Fusion — merge steps 3 & 4
    6. Ethical Evaluation — multi-philosophy EDS scoring
    7. Decision Gate      — approve / escalate / reject / hard-block
    8. Reflection          — post-decision learning & audit

Stages 3 and 4 are logically independent (could be parallelized in
a future async rewrite — for now they run sequentially but their
outputs are merged in stage 5).

Every stage writes a StageResult into PipelineState, so the full
decision trace is always available for explainability and audit.

Author notes
------------
This file has grown a lot since the first prototype.  The retry
logic (exponential back-off for flaky LLM calls) was added after
a demo where GPT-4 rate-limited us mid-batch.  The ``dry_run``
flag was a reviewer request so they could inspect the pipeline
without burning API credits.

"""

from __future__ import annotations

import logging
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ethicagent.agents.action_executor import ActionExecutor
from ethicagent.agents.context_agent import ContextAgent
from ethicagent.agents.ethical_reasoner import EthicalReasonerAgent
from ethicagent.agents.fusion_agent import FusionAgent
from ethicagent.agents.human_gateway import HumanGateway
from ethicagent.agents.neural_reasoner import NeuralReasoner
from ethicagent.agents.reflection_agent import ReflectionAgent
from ethicagent.agents.symbolic_reasoner import SymbolicReasoner
from ethicagent.core.logger import AuditLogger
from ethicagent.core.state import (
    PipelineStage,
    PipelineState,
    StateManager,
)
from ethicagent.knowledge.knowledge_graph import KnowledgeGraph
from ethicagent.knowledge.memory_store import MemoryStore
from ethicagent.knowledge.precedent_store import PrecedentStore
from ethicagent.utils.config_loader import ConfigLoader
from ethicagent.utils.helpers import now_iso
from ethicagent.utils.validators import validate_task_input

logger = logging.getLogger(__name__)

# -- small constants used throughout -----------------------------------------------
_MAX_LLM_RETRIES = 3
_BACKOFF_BASE = 1.5  # exponential back-off base (seconds)
_BACKOFF_JITTER = 0.3  # ±jitter to avoid thundering-herd


@dataclass
class PipelineResult:
    """Rich result object returned by ``EthicAgentOrchestrator.run()``.

    Stores everything a downstream consumer might need — the final
    verdict *and* every intermediate stage output, so the decision
    trace can be replayed or audited later.

    We intentionally keep ``stage_outputs`` as a plain dict (stage
    name → output dict) rather than a typed hierarchy, because the
    data shape varies per stage and we don't want to force callers
    into our internal types.
    """

    action_id: str
    status: str  # "completed" | "error" | …
    domain: str = "general"
    eds_score: float = 0.0
    verdict: str = "unknown"
    confidence: float = 0.0
    philosophy_scores: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    rules_triggered: list[str] = field(default_factory=list)
    conflict_analysis: dict[str, Any] = field(default_factory=dict)
    execution_result: dict[str, Any] = field(default_factory=dict)
    reflection: dict[str, Any] = field(default_factory=dict)
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)
    total_elapsed_s: float = 0.0
    timestamp: str = field(default_factory=now_iso)
    error: str | None = None

    # -- pretty-printing -------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"PipelineResult(action={self.action_id!r}, "
            f"verdict={self.verdict!r}, eds={self.eds_score:.3f}, "
            f"elapsed={self.total_elapsed_s:.2f}s)"
        )

    def __str__(self) -> str:
        lines = [
            "╔══ EthicAgent Pipeline Result ═══════════════",
            f"║ Action   : {self.action_id}",
            f"║ Domain   : {self.domain}",
            f"║ Verdict  : {self.verdict.upper()}",
            f"║ EDS Score: {self.eds_score:.4f}",
            f"║ Confidence: {self.confidence:.2%}",
            f"║ Elapsed  : {self.total_elapsed_s:.3f}s",
        ]
        if self.philosophy_scores:
            lines.append("║ Philosophy scores:")
            for name, sc in self.philosophy_scores.items():
                lines.append(f"║   {name:20s}: {sc:.3f}")
        if self.rules_triggered:
            lines.append(f"║ Rules triggered: {len(self.rules_triggered)}")
        lines.append("╚═════════════════════════════════════════════")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Flatten everything into a JSON-friendly dict."""
        return {
            "action_id": self.action_id,
            "status": self.status,
            "domain": self.domain,
            "eds_score": self.eds_score,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "philosophy_scores": self.philosophy_scores,
            "weights_used": self.weights_used,
            "reasoning": self.reasoning,
            "rules_triggered": self.rules_triggered,
            "conflict_analysis": self.conflict_analysis,
            "execution_result": self.execution_result,
            "reflection": self.reflection,
            "stage_outputs": self.stage_outputs,
            "stage_timings": self.stage_timings,
            "total_elapsed_s": self.total_elapsed_s,
            "timestamp": self.timestamp,
            "error": self.error,
        }


class EthicAgentOrchestrator:
    """Main orchestrator for the EthicAgent pipeline.

    Coordinates all agents, knowledge stores, and evaluation modules
    through an 8-step ethical reasoning pipeline with full traceability.

    Usage::

        orch = EthicAgentOrchestrator(config_dir="config/")
        result = orch.run("Deny insulin to a diabetic patient", domain="healthcare")
        print(result)  # pretty-printed verdict

    The orchestrator is deliberately synchronous for simplicity —
    concurrent execution (e.g. for web APIs) should run separate
    instances behind a task queue.
    """

    def __init__(
        self,
        config_dir: str | None = None,
        review_callback: Callable | None = None,
        use_llm: bool = True,
        llm_provider: str = "openai",
    ) -> None:
        """Wire up the full pipeline.

        Args:
            config_dir:      Path to the YAML config directory.
            review_callback: Sync callback for human-in-the-loop review.
            use_llm:         Set ``False`` to skip the Neural Reasoner
                             and rely entirely on heuristics + rules.
            llm_provider:    "openai" or "ollama".
        """
        logger.info("Initializing EthicAgent orchestrator …")

        # -- configuration --------------------------------------------------------
        if config_dir:
            from pathlib import Path

            cfg_path = Path(config_dir)
            if cfg_path.is_file():
                self.config = ConfigLoader.from_file(cfg_path)
            elif cfg_path.is_dir():
                # Try loading individual config files from the directory
                self.config = ConfigLoader()
                for name in ("ethical_rules", "domain_weights", "llm_config"):
                    for ext in (".yaml", ".yml", ".json"):
                        f = cfg_path / f"{name}{ext}"
                        if f.exists():
                            sub = ConfigLoader.from_file(f)
                            self.config.set(name, sub.to_dict())
                            break
            else:
                self.config = ConfigLoader()
        else:
            self.config = ConfigLoader()
        self.ethical_rules = self.config.get("ethical_rules", {})
        self.domain_weights_cfg = self.config.get("domain_weights", {})
        self.llm_config = self.config.get("llm_config", {})

        # -- core infra -----------------------------------------------------------
        self.state_manager = StateManager()
        self.audit_logger = AuditLogger()

        # -- knowledge stores -----------------------------------------------------
        self.knowledge_graph = KnowledgeGraph()
        self.precedent_store = PrecedentStore()
        # ChromaDB can be flaky in CI — fall back gracefully
        try:
            self.memory_store: MemoryStore | None = MemoryStore()
        except Exception as exc:
            logger.warning(
                "MemoryStore init failed — running without semantic memory. "
                "This is fine for testing but you'll lose precedent-based "
                f"retrieval.  Error: {exc}"
            )
            self.memory_store = None

        # -- domain weights -------------------------------------------------------
        domain_weights = self._extract_domain_weights()

        # -- agents ---------------------------------------------------------------
        self.context_agent = ContextAgent(
            knowledge_graph=self.knowledge_graph,
        )
        self.neural_reasoner = NeuralReasoner(
            config=self.llm_config,
            use_llm=use_llm,
            provider=llm_provider,
        )
        self.symbolic_reasoner = SymbolicReasoner(
            rules=self.ethical_rules,
            knowledge_graph=self.knowledge_graph,
        )
        self.fusion_agent = FusionAgent()
        self.ethical_reasoner = EthicalReasonerAgent(
            rules=self.ethical_rules,
            domain_weights=domain_weights,
        )
        self.action_executor = ActionExecutor(
            audit_logger=self.audit_logger,
        )
        self.human_gateway = HumanGateway(
            review_callback=review_callback,
        )
        self.reflection_agent = ReflectionAgent(
            precedent_store=self.precedent_store,
            memory_store=self.memory_store,
        )

        # -- bookkeeping ----------------------------------------------------------
        self._total_runs: int = 0
        self._total_time: float = 0.0
        self._verdict_counts: dict[str, int] = {}

        logger.info("Orchestrator ready ✓")

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  PUBLIC API                                                  ║
    # ╚══════════════════════════════════════════════════════════════╝

    def run(
        self,
        task: str,
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Run the full 8-stage ethical reasoning pipeline.

        Args:
            task:     Free-text description of the action to evaluate.
            domain:   Optional domain hint (auto-detected otherwise).
            metadata: Extra key-value pairs forwarded to the context agent.
            dry_run:  If True, trace the pipeline stages without actually
                      calling the LLM or executing the decision.  Useful
                      for pipeline inspection and testing.

        Returns:
            A ``PipelineResult`` with the verdict, scores, and full
            decision trace.
        """
        t_start = time.time()
        action_id = f"ACT-{uuid.uuid4().hex[:8].upper()}"
        self._total_runs += 1
        stage_timings: dict[str, float] = {}
        stage_outputs: dict[str, Any] = {}

        logger.info(
            f"Pipeline started: action_id={action_id}, "
            f"task={task[:60]!r}{'…' if len(task) > 60 else ''}"
        )

        # -- validate input -------------------------------------------------------
        validation = validate_task_input(task)
        if not validation["valid"]:
            return PipelineResult(
                action_id=action_id,
                status="validation_error",
                error="; ".join(validation.get("errors", ["invalid input"])),
            )

        # -- create shared state ---------------------------------------------------
        state = self.state_manager.create_state(task=task, domain=domain or "general")

        pipeline_input = {
            "task": task,
            "action_id": action_id,
            "domain": domain,
            "metadata": metadata or {},
        }

        if dry_run:
            return self._dry_run(action_id, pipeline_input, state)

        try:
            # ── Stage 1: Context Extraction ──────────────────────────
            ctx, dt = self._timed(self._stage_context, state, pipeline_input)
            stage_timings["context_extraction"] = dt
            stage_outputs["context"] = ctx

            detected_domain = domain or ctx.get("domain", "general")

            # ── Stage 2: Knowledge Retrieval ─────────────────────────
            kg_out, dt = self._timed(self._stage_knowledge, state, detected_domain)
            stage_timings["knowledge_retrieval"] = dt
            stage_outputs["knowledge"] = kg_out

            # ── Stage 3: Neural Reasoning ────────────────────────────
            # Wrapped in retry logic — LLM APIs can be flaky
            neural_out, dt = self._timed(self._stage_neural, state, ctx, detected_domain)
            stage_timings["neural_reasoning"] = dt
            stage_outputs["neural"] = neural_out

            # ── Stage 4: Symbolic Reasoning ──────────────────────────
            sym_out, dt = self._timed(self._stage_symbolic, state, ctx, detected_domain)
            stage_timings["symbolic_reasoning"] = dt
            stage_outputs["symbolic"] = sym_out

            # ── Stage 5: Fusion ──────────────────────────────────────
            fused, dt = self._timed(self._stage_fusion, state, neural_out, sym_out, detected_domain)
            stage_timings["fusion"] = dt
            stage_outputs["fusion"] = fused

            # ── Stage 6: Ethical Evaluation (the core!) ──────────────
            decision, dt = self._timed(self._stage_ethical_eval, state, ctx, fused, detected_domain)
            stage_timings["ethical_evaluation"] = dt
            stage_outputs["ethical_decision"] = {
                "eds_score": decision.eds_score,
                "verdict": decision.verdict.value,
                "confidence": decision.confidence,
                "philosophy_scores": {pr.name: pr.score for pr in decision.philosophy_results},
            }

            # ── Stage 7: Decision Gate ───────────────────────────────
            exec_result, dt = self._timed(self._stage_decision_gate, state, decision, ctx)
            stage_timings["decision_gate"] = dt
            stage_outputs["execution"] = exec_result

            # Handle escalation (human-in-the-loop)
            if exec_result.get("status") == "escalated":
                review = self.human_gateway.escalate(
                    decision,
                    ctx,
                    priority=exec_result.get("review_priority", "medium"),
                )
                exec_result["human_review"] = review

            # ── Stage 8: Reflection ──────────────────────────────────
            refl, dt = self._timed(self._stage_reflection, state, decision, ctx, exec_result)
            stage_timings["reflection"] = dt
            stage_outputs["reflection"] = refl

            # mark pipeline done
            state.update_stage(PipelineStage.COMPLETED, {"status": "ok"})

        except Exception as exc:
            logger.error(f"Pipeline failed: {exc}", exc_info=True)
            elapsed = time.time() - t_start
            return PipelineResult(
                action_id=action_id,
                status="pipeline_error",
                error=str(exc),
                total_elapsed_s=elapsed,
                stage_timings=stage_timings,
                stage_outputs=stage_outputs,
            )

        elapsed = time.time() - t_start
        self._total_time += elapsed

        verdict_str = decision.verdict.value
        self._verdict_counts[verdict_str] = self._verdict_counts.get(verdict_str, 0) + 1

        result = PipelineResult(
            action_id=action_id,
            status="completed",
            domain=detected_domain,
            eds_score=decision.eds_score,
            verdict=verdict_str,
            confidence=decision.confidence,
            philosophy_scores={pr.name: pr.score for pr in decision.philosophy_results},
            weights_used=decision.weights_used,
            reasoning=decision.reasoning,
            rules_triggered=decision.rules_triggered,
            conflict_analysis=decision.conflict_analysis,
            execution_result=exec_result,
            reflection=refl,
            stage_outputs=stage_outputs,
            stage_timings=stage_timings,
            total_elapsed_s=elapsed,
        )

        logger.info(
            f"Pipeline complete: {action_id} → {verdict_str.upper()} "
            f"(EDS={decision.eds_score:.3f}, {elapsed:.2f}s)"
        )
        return result

    def run_batch(
        self,
        tasks: list[dict[str, Any]],
        progress_callback: Callable[[int, int, PipelineResult], None] | None = None,
        *,
        dry_run: bool = False,
    ) -> list[PipelineResult]:
        """Evaluate a list of tasks through the pipeline.

        Each entry is a dict with keys ``task``, and optionally
        ``domain`` and ``metadata``.

        Args:
            tasks:             List of task dicts.
            progress_callback: ``fn(current_idx, total, last_result)``
                               called after each task.
            dry_run:           Forward to ``run()``.

        Returns:
            List of PipelineResult in the same order as *tasks*.
        """
        results: list[PipelineResult] = []
        total = len(tasks)
        logger.info(f"Starting batch evaluation ({total} tasks)")

        for idx, t_input in enumerate(tasks, start=1):
            res = self.run(
                task=t_input.get("task", ""),
                domain=t_input.get("domain"),
                metadata=t_input.get("metadata"),
                dry_run=dry_run,
            )
            results.append(res)
            if progress_callback:
                progress_callback(idx, total, res)
            if idx % 25 == 0 or idx == total:
                # periodic progress log so long batches aren't silent
                logger.info(f"  batch progress: {idx}/{total}")

        logger.info(f"Batch complete — {total} tasks processed")
        return results

    def get_statistics(self) -> dict[str, Any]:
        """Return aggregate pipeline statistics."""
        return {
            "total_runs": self._total_runs,
            "total_time_seconds": round(self._total_time, 3),
            "avg_time_seconds": round(self._total_time / max(self._total_runs, 1), 3),
            "verdict_distribution": dict(self._verdict_counts),
            "precedent_store": self.precedent_store.get_statistics(),
            "review_stats": self.human_gateway.get_review_statistics(),
            "reflection_summary": self.reflection_agent.get_learning_summary(),
        }

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  STAGE IMPLEMENTATIONS                                       ║
    # ╚══════════════════════════════════════════════════════════════╝

    def _stage_context(self, state: PipelineState, inp: dict[str, Any]) -> dict[str, Any]:
        """Stage 1 — Context Extraction.

        The ContextAgent pulls out domain, entities, stakeholders,
        urgency, and other signals from the free-text task description.
        """
        state.update_stage(PipelineStage.CONTEXT_EXTRACTION, {"status": "running"})

        ctx = self.context_agent.extract_context(
            inp.get("task", ""),
            domain_hint=inp.get("domain"),
            metadata=inp.get("metadata", {}),
        )
        ctx["action_id"] = inp["action_id"]

        state.update_stage(
            PipelineStage.CONTEXT_EXTRACTION,
            {
                "domain": ctx.get("domain", "general"),
                "entities_found": len(ctx.get("entities", [])),
            },
        )
        return ctx

    def _stage_knowledge(self, state: PipelineState, domain: str) -> dict[str, Any]:
        """Stage 2 — Knowledge Retrieval from the domain graph."""
        state.update_stage(PipelineStage.KNOWLEDGE_QUERY, {"status": "running"})

        rules = self.knowledge_graph.get_applicable_laws(domain)
        protected = self.knowledge_graph.get_protected_attributes(domain)

        out = {
            "rules_loaded": len(rules),
            "applicable_laws": rules,
            "protected_attributes": protected,
            "domain": domain,
        }
        state.update_stage(PipelineStage.KNOWLEDGE_QUERY, out)
        return out

    def _stage_neural(
        self, state: PipelineState, ctx: dict[str, Any], domain: str
    ) -> dict[str, Any]:
        """Stage 3 — Neural (LLM) reasoning with retry + back-off.

        We retry up to 3 times with exponential back-off because
        the OpenAI API occasionally returns 429s or 503s during peak
        hours.  If all retries fail the NeuralReasoner's own heuristic
        fallback kicks in (see neural_reasoner.py).
        """
        state.update_stage(PipelineStage.NEURAL_REASONING, {"status": "running"})

        for attempt in range(_MAX_LLM_RETRIES):
            try:
                result = self.neural_reasoner.reason(ctx, domain)
                state.update_stage(
                    PipelineStage.NEURAL_REASONING,
                    {
                        "recommendation": result.get("recommendation"),
                        "confidence": result.get("confidence"),
                        "source": result.get("source", "unknown"),
                    },
                )
                return result
            except Exception as exc:
                wait = _BACKOFF_BASE**attempt + random.uniform(-_BACKOFF_JITTER, _BACKOFF_JITTER)
                logger.warning(
                    f"Neural reasoning attempt {attempt + 1}/{_MAX_LLM_RETRIES} "
                    f"failed: {exc}. Retrying in {wait:.1f}s …"
                )
                time.sleep(wait)

        # all retries exhausted — let the reasoner's heuristic handle it
        logger.error("Neural reasoning failed after all retries — using heuristic fallback")
        result = self.neural_reasoner.reason(ctx, domain)
        return result

    def _stage_symbolic(
        self, state: PipelineState, ctx: dict[str, Any], domain: str
    ) -> dict[str, Any]:
        """Stage 4 — Symbolic (rule-based) reasoning."""
        state.update_stage(PipelineStage.SYMBOLIC_REASONING, {"status": "running"})
        result = self.symbolic_reasoner.reason(ctx, domain)
        state.update_stage(
            PipelineStage.SYMBOLIC_REASONING,
            {
                "status": result.get("status"),
                "blocked": result.get("blocked", False),
                "rules_matched": len(result.get("matched_rules", [])),
            },
        )
        return result

    def _stage_fusion(
        self,
        state: PipelineState,
        neural: dict[str, Any],
        symbolic: dict[str, Any],
        domain: str,
    ) -> dict[str, Any]:
        """Stage 5 — Fuse neural + symbolic outputs.

        The safety-first principle:  if the symbolic reasoner says
        'block' we respect that, even if the LLM thinks it's fine.
        This was a hard lesson from early testing.
        """
        state.update_stage(PipelineStage.FUSION, {"status": "running"})
        result = self.fusion_agent.fuse(neural, symbolic, domain)
        state.update_stage(
            PipelineStage.FUSION,
            {
                "recommendation": result.get("recommendation"),
                "agreement": result.get("agreement"),
            },
        )
        return result

    def _stage_ethical_eval(
        self,
        state: PipelineState,
        ctx: dict[str, Any],
        fusion: dict[str, Any],
        domain: str,
    ):
        """Stage 6 — Multi-philosophy ethical evaluation (EDS formula)."""
        state.update_stage(PipelineStage.ETHICAL_EVALUATION, {"status": "running"})
        decision = self.ethical_reasoner.evaluate(ctx, fusion, domain)
        state.update_stage(
            PipelineStage.ETHICAL_EVALUATION,
            {
                "eds_score": decision.eds_score,
                "verdict": decision.verdict.value,
                "confidence": decision.confidence,
            },
        )
        return decision

    def _stage_decision_gate(
        self, state: PipelineState, decision, ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Stage 7 — Decision gate + action execution."""
        state.update_stage(PipelineStage.DECISION_GATE, {"status": "running"})
        result = self.action_executor.execute(decision, ctx)
        state.update_stage(
            PipelineStage.DECISION_GATE,
            {
                "action_status": result.get("status"),
                "requires_review": result.get("requires_human_review", False),
            },
        )
        return result

    def _stage_reflection(
        self,
        state: PipelineState,
        decision,
        ctx: dict[str, Any],
        exec_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Stage 8 — Post-decision reflection & learning."""
        state.update_stage(PipelineStage.REFLECTION, {"status": "running"})
        refl = self.reflection_agent.reflect(decision, ctx, exec_result)
        state.update_stage(
            PipelineStage.REFLECTION,
            {
                "recommendations": len(refl.get("recommendations", [])),
                "is_consistent": refl.get("consistency_analysis", {}).get("is_consistent"),
            },
        )
        return refl

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  INTERNAL HELPERS                                            ║
    # ╚══════════════════════════════════════════════════════════════╝

    @staticmethod
    def _timed(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Call *fn* and return ``(result, elapsed_seconds)``."""
        t0 = time.time()
        result = fn(*args, **kwargs)
        return result, time.time() - t0

    def _dry_run(
        self,
        action_id: str,
        inp: dict[str, Any],
        state: PipelineState,
    ) -> PipelineResult:
        """Trace the pipeline stages without executing anything.

        Useful for debugging or inspecting the pipeline configuration
        without burning LLM tokens.
        """
        stages_traced = [
            "context_extraction",
            "knowledge_retrieval",
            "neural_reasoning (SKIPPED — dry_run)",
            "symbolic_reasoning",
            "fusion",
            "ethical_evaluation",
            "decision_gate",
            "reflection",
        ]
        logger.info(f"[dry_run] Pipeline traced: {stages_traced}")

        return PipelineResult(
            action_id=action_id,
            status="dry_run",
            domain=inp.get("domain", "general"),
            stage_outputs={"stages_traced": stages_traced, "input": inp},
        )

    def _extract_domain_weights(self) -> dict[str, dict[str, float]]:
        """Pull per-domain philosophy weights from config.

        Falls back to uniform weights (0.25 each) when a domain
        isn't explicitly configured.

        Supports two YAML formats:
          - Named keys: ``weights: {deontological: 0.35, ...}``
          - Short keys: ``w1: 0.35, w2: 0.25, w3: 0.20, w4: 0.20``
        """
        weights: dict[str, dict[str, float]] = {}
        domains_cfg = self.domain_weights_cfg.get("domains", {})
        if not isinstance(domains_cfg, dict):
            domains_cfg = {}

        for dom_name, dom_cfg in domains_cfg.items():
            if not isinstance(dom_cfg, dict):
                # Skip non-dict entries (e.g. "default": "general")
                continue

            # Support both formats: nested "weights" dict or flat w1/w2/w3/w4
            w = dom_cfg.get("weights", {})
            if w and isinstance(w, dict):
                weights[dom_name] = {
                    "deontological": w.get("deontological", 0.25),
                    "consequentialist": w.get("consequentialist", 0.25),
                    "virtue_ethics": w.get("virtue_ethics", w.get("virtue", 0.25)),
                    "contextual": w.get("contextual", 0.25),
                }
            elif "w1" in dom_cfg:
                # Short-form keys from domain_weights.yaml
                weights[dom_name] = {
                    "deontological": float(dom_cfg.get("w1", 0.25)),
                    "consequentialist": float(dom_cfg.get("w2", 0.25)),
                    "virtue_ethics": float(dom_cfg.get("w3", 0.25)),
                    "contextual": float(dom_cfg.get("w4", 0.25)),
                }
            # else: no recognisable weight keys → skip, general fallback applies

        # always have a general fallback
        if "general" not in weights:
            weights["general"] = {
                "deontological": 0.25,
                "consequentialist": 0.25,
                "virtue_ethics": 0.25,
                "contextual": 0.25,
            }

        return weights

    # -- dunder helpers --------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"EthicAgentOrchestrator(runs={self._total_runs}, "
            f"avg={self._total_time / max(self._total_runs, 1):.2f}s)"
        )

    def __str__(self) -> str:
        return (
            f"EthicAgent Orchestrator — {self._total_runs} runs completed, "
            f"avg {self._total_time / max(self._total_runs, 1):.2f}s/run"
        )


def main() -> None:
    """CLI entry point for ethicagent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EthicAgent — Neuro-Symbolic Ethical Reasoning CLI",
    )
    parser.add_argument("task", nargs="?", default=None, help="Task to evaluate")
    parser.add_argument(
        "--domain", default=None, help="Domain hint (healthcare, finance, hiring, disaster)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (skip LLM calls)")
    args = parser.parse_args()

    if not args.task:
        print("EthicAgent v1.0.0 — Neuro-Symbolic Ethical Reasoning")
        print("Usage: ethicagent 'Approve loan for applicant' --domain finance")
        return

    orch = EthicAgentOrchestrator(use_llm=False)
    result = orch.run(args.task, domain=args.domain, dry_run=args.dry_run)
    print(result)
