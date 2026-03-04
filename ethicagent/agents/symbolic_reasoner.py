"""Symbolic Reasoner — rule-based constraint checking (Stage 4).

Evaluates the action against a set of hard rules, duty rules, and
domain-specific rules loaded from config.  Supports:

  • exact and fuzzy keyword matching
  • rule chaining (if rule A fires, also check dependent rule B)
  • rule priority levels (critical > high > medium > low)
  • negation (NOT conditions)
  • explanation generation per matched rule
  • full trace mode for debugging

The rule engine is inspired by Prolog's backward chaining but a lot
simpler — we don't do unification, just pattern matching.

# HACK: fuzzy matching uses simple substring + Levenshtein-lite.
#       For production, consider a proper rule engine (Drools, etc.)
# NOTE: rule chaining was added after discovering that 'deny treatment'
#       should also trigger 'patient rights violation'.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class SymbolicReasoner:
    """Rule-based ethical constraint engine.

    Evaluates actions against hard rules (binary pass/fail), duty rules
    (graduated compliance), and domain rules (context-specific).

    The ``reason()`` method returns a dict with matched rules, overall
    status, and whether the action should be blocked.
    """

    # -- rule priority ordering -----------------------------------------------
    PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def __init__(
        self,
        rules: dict[str, Any] | None = None,
        knowledge_graph: Any = None,
    ) -> None:
        cfg = rules or {}
        self._hard_rules = cfg.get("hard_rules", [])
        self._duty_rules = cfg.get("duty_rules", [])
        self._domain_rules = cfg.get("domain_rules", {})
        self._cultural_rules = cfg.get("cultural_sensitivity_rules", [])
        self._legal_rules = cfg.get("legal_compliance", [])
        self._kg = knowledge_graph

        # -- rule chain definitions -------------------------------------------
        # If rule X fires, also evaluate rules in its "triggers" list.
        # Built from config or hardcoded fallback.
        self._chains: dict[str, list[str]] = cfg.get("rule_chains", {})

        logger.info(
            f"SymbolicReasoner loaded: {len(self._hard_rules)} hard rules, "
            f"{len(self._duty_rules)} duty rules, "
            f"{len(self._domain_rules)} domain-specific rule sets"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(
        self,
        context: dict[str, Any],
        domain: str,
        *,
        trace: bool = False,
    ) -> dict[str, Any]:
        """Evaluate all rules against the given context.

        Args:
            context: Enriched context from ContextAgent.
            domain:  Classified domain.
            trace:   If True, include detailed reasoning chain.

        Returns:
            Dict with keys: status, blocked, matched_rules,
            hard_violations, duty_scores, explanations, trace.
        """
        task = context.get("action", "")
        matched: list[dict[str, Any]] = []
        explanations: list[str] = []
        hard_violations: list[str] = []
        blocked = False
        trace_log: list[str] = []

        # -- 1. Hard rules (binary: violated or not) -------------------------
        for rule in self._hard_rules:
            hit, explanation = self._check_rule(rule, task, context, domain)
            if trace:
                trace_log.append(f"HARD {rule.get('id', '?')}: {'HIT' if hit else 'miss'}")
            if hit:
                matched.append({**rule, "type": "hard", "violated": True})
                hard_violations.append(rule.get("id", "unknown"))
                explanations.append(explanation)
                blocked = True
                logger.warning(
                    f"Hard rule violated: {rule.get('id')} — {rule.get('description', '')[:80]}"
                )

        # -- 2. Duty rules (graduated compliance) ----------------------------
        duty_scores: dict[str, float] = {}
        for rule in self._duty_rules:
            hit, explanation = self._check_rule(rule, task, context, domain)
            rid = rule.get("id", "duty")
            if trace:
                trace_log.append(f"DUTY {rid}: {'HIT' if hit else 'miss'}")
            if hit:
                # duty rule hit → lower compliance score
                severity = rule.get("severity", 0.3)
                duty_scores[rid] = 1.0 - severity
                matched.append({**rule, "type": "duty", "compliance": 1.0 - severity})
                explanations.append(explanation)

        # -- 3. Domain-specific rules ----------------------------------------
        dom_rules = self._domain_rules.get(domain, [])
        for rule in dom_rules:
            hit, explanation = self._check_rule(rule, task, context, domain)
            rid = rule.get("id", "dom")
            if trace:
                trace_log.append(f"DOMAIN({domain}) {rid}: {'HIT' if hit else 'miss'}")
            if hit:
                matched.append({**rule, "type": "domain"})
                explanations.append(explanation)

        # -- 4. Cultural & legal rules ---------------------------------------
        for rule in self._cultural_rules + self._legal_rules:
            hit, explanation = self._check_rule(rule, task, context, domain)
            if hit:
                matched.append({**rule, "type": rule.get("type", "legal")})
                explanations.append(explanation)

        # -- 5. Rule chaining ------------------------------------------------
        chained = self._apply_chains(matched, task, context, domain, trace_log)
        matched.extend(chained)

        # -- 6. KG constraints (if available) ---------------------------------
        kg_issues: list[str] = []
        if self._kg:
            try:
                # Check if any protected attributes are being used in this domain
                protected = self._kg.get_protected_attributes(domain)
                for attr in protected:
                    if attr in task.lower():
                        kg_issues.append(
                            f"Protected attribute '{attr}' must not influence {domain} decisions"
                        )
                        explanations.append(
                            f"Knowledge graph constraint: protected attribute '{attr}'"
                        )
            except Exception as exc:
                logger.debug(f"KG constraint check failed: {exc}")

        # Sort matched rules by priority
        matched.sort(key=lambda r: self.PRIORITY_ORDER.get(r.get("priority", "medium"), 2))

        # Compute overall symbolic score (0–1)
        if blocked:
            sym_score = 0.0
        elif duty_scores:
            sym_score = sum(duty_scores.values()) / len(duty_scores)
        else:
            sym_score = 1.0  # no rules triggered → fully compliant

        status = "blocked" if blocked else ("caution" if matched else "compliant")

        result: dict[str, Any] = {
            "status": status,
            "blocked": blocked,
            "symbolic_score": round(sym_score, 3),
            "matched_rules": matched,
            "hard_violations": hard_violations,
            "duty_scores": duty_scores,
            "explanations": explanations,
            "kg_issues": kg_issues,
        }
        if trace:
            result["trace"] = trace_log

        logger.info(
            f"Symbolic reasoning: status={status}, matched={len(matched)} rules, blocked={blocked}"
        )
        return result

    # ------------------------------------------------------------------
    # Rule checking engine
    # ------------------------------------------------------------------

    def _check_rule(
        self,
        rule: dict[str, Any],
        task: str,
        context: dict[str, Any],
        domain: str,
    ) -> tuple[bool, str]:
        """Check whether a single rule matches.

        Returns (matched: bool, explanation: str).
        Supports: keyword matching, condition expressions (AND/OR/NOT),
        fuzzy matching, and domain filtering.
        """
        # -- domain scope check -----------------------------------------------
        rule_domains = rule.get("domains", [])
        if rule_domains and domain not in rule_domains and "all" not in rule_domains:
            return False, ""

        task_lower = task.lower()

        # -- keyword matching (with fuzzy option) -----------------------------
        keywords = rule.get("keywords", [])
        conditions = rule.get("conditions", [])
        fuzzy = rule.get("fuzzy", False)

        # keyword check
        kw_hit = False
        if keywords:
            if fuzzy:
                kw_hit = any(self._fuzzy_match(kw.lower(), task_lower) for kw in keywords)
            else:
                kw_hit = any(kw.lower() in task_lower for kw in keywords)

        # -- condition expressions (AND, OR, NOT) -----------------------------
        cond_hit = True
        if conditions:
            cond_hit = self._evaluate_conditions(conditions, task, context)

        # final hit: keywords AND conditions both must be satisfied
        # (if either list is empty, treat as True)
        hit = True
        if keywords:
            hit = hit and kw_hit
        if conditions:
            hit = hit and cond_hit

        # also try pattern matching if rule has a 'pattern' field
        # Pattern acts as an *additional* standalone check only when
        # no keyword/condition gates were defined.  When keywords or
        # conditions are present, the pattern is an extra required AND
        # signal rather than an unconditional override.
        pattern = rule.get("pattern")
        if pattern:
            try:
                pattern_hit = bool(re.search(pattern, task, re.IGNORECASE))
            except re.error:
                pattern_hit = False
            if not keywords and not conditions:
                # No keyword/condition gates → pattern alone decides
                hit = pattern_hit
            else:
                # Pattern is an extra AND requirement
                hit = hit and pattern_hit

        explanation = ""
        if hit:
            desc = rule.get("description", rule.get("id", "rule"))
            explanation = f"Rule {rule.get('id', '?')}: {desc}"

        return hit, explanation

    def _evaluate_conditions(
        self,
        conditions: list[Any],
        task: str,
        context: dict[str, Any],
    ) -> bool:
        """Evaluate a list of condition expressions.

        Supports string conditions with simple operators:
          "urgency == emergency"
          "NOT routine"
          "domain == healthcare AND urgency != low"
        """
        task_lower = task.lower()

        for cond in conditions:
            if isinstance(cond, str):
                cond_str = cond.strip()

                # -- NOT prefix -----------------------------------------------
                if cond_str.upper().startswith("NOT "):
                    inner = cond_str[4:].strip().lower()
                    if inner in task_lower:
                        return False
                    continue

                # -- key == value check from context --------------------------
                if "==" in cond_str:
                    key, val = [s.strip() for s in cond_str.split("==", 1)]
                    ctx_val = str(context.get(key, "")).lower()
                    if ctx_val != val.lower():
                        return False
                    continue

                # -- key != value ---------------------------------------------
                if "!=" in cond_str:
                    key, val = [s.strip() for s in cond_str.split("!=", 1)]
                    ctx_val = str(context.get(key, "")).lower()
                    if ctx_val == val.lower():
                        return False
                    continue

                # -- plain keyword check --------------------------------------
                if cond_str.lower() not in task_lower:
                    return False

        return True

    @staticmethod
    def _fuzzy_match(pattern: str, text: str, threshold: int = 80) -> bool:
        """Fuzzy matching using proper Levenshtein distance via *rapidfuzz*.

        Uses token-level partial matching so multi-word patterns
        (e.g. ``"deny treatment"``) are matched as phrases rather
        than individual independent words.

        Args:
            pattern:   The keyword/phrase to search for.
            text:      The text to search within.
            threshold: Minimum similarity score (0-100) to count as a
                       match.  Default 80 balances recall vs. precision.

        Returns:
            True if *pattern* appears in *text* with at least
            *threshold* similarity.
        """
        if pattern in text:
            return True

        try:
            from rapidfuzz import fuzz

            # partial_ratio handles substring matching well
            score = fuzz.partial_ratio(pattern, text)
            return score >= threshold
        except ImportError:
            # Fallback: basic character-level comparison (original logic)
            p_words = pattern.split()
            t_words = text.split()
            for pw in p_words:
                if any(
                    abs(len(pw) - len(tw)) <= 2
                    and sum(a != b for a, b in zip(pw, tw, strict=False)) <= 2
                    for tw in t_words
                ):
                    return True
            return False

    def _apply_chains(
        self,
        already_matched: list[dict[str, Any]],
        task: str,
        context: dict[str, Any],
        domain: str,
        trace_log: list[str],
    ) -> list[dict[str, Any]]:
        """If a matched rule has dependent rules, evaluate those too.

        Prevents infinite loops by tracking seen rule IDs.
        """
        extra: list[dict[str, Any]] = []
        seen: set[str] = {r.get("id", "") for r in already_matched}

        for r in already_matched:
            rid = r.get("id", "")
            chain_targets = self._chains.get(rid, [])
            for target_id in chain_targets:
                if target_id in seen:
                    continue
                seen.add(target_id)
                # find the target rule in all rule lists
                target_rule = self._find_rule_by_id(target_id)
                if target_rule:
                    hit, expl = self._check_rule(target_rule, task, context, domain)
                    trace_log.append(f"CHAIN {rid} → {target_id}: {'HIT' if hit else 'miss'}")
                    if hit:
                        extra.append({**target_rule, "type": "chained"})
        return extra

    def _find_rule_by_id(self, rule_id: str) -> dict[str, Any] | None:
        """Search all rule lists for a rule with the given ID."""
        for pool in [
            self._hard_rules,
            self._duty_rules,
            self._cultural_rules,
            self._legal_rules,
        ]:
            for r in pool:
                if r.get("id") == rule_id:
                    return r
        for dom_rules in self._domain_rules.values():
            for r in dom_rules:
                if r.get("id") == rule_id:
                    return r
        return None
