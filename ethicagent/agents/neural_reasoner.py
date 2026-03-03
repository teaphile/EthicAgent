"""Neural Reasoner — LLM-based ethical reasoning (Stage 3).

Sends a structured prompt to an LLM (OpenAI GPT-4 by default, with
Ollama as fallback) and parses the response into a recommendation
dict.  If the LLM is unavailable or disabled, drops to a heuristic
fallback that does keyword-analysis + rule-matching + simple sentiment.

The heuristic isn't great but it's deterministic and offline, which
is perfect for CI and unit tests.  For the paper results we always
use GPT-4 (temperature=0.2 — tried 0.7 first but it gave wildly
inconsistent ethical reasoning).

# NOTE: chain-of-thought (CoT) prompting improved reasoning quality
#       noticeably in our ablation — see the prompt templates below.
# TODO: try structured output (function calling) for more reliable parsing
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -- LRU-ish cache for LLM responses -----------------------------------------
# We hash (prompt_text + model) and keep the last N responses so repeated
# evaluations of the same scenario don't burn tokens.
_CACHE_MAX = 256


class _ResponseCache:
    """Simple LRU dict for caching LLM responses by prompt hash."""

    def __init__(self, maxsize: int = _CACHE_MAX) -> None:
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()[:16]

    def get(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        k = self._key(prompt, model)
        if k in self._store:
            self._store.move_to_end(k)
            self.hits += 1
            return self._store[k]
        self.misses += 1
        return None

    def put(self, prompt: str, model: str, value: Dict[str, Any]) -> None:
        k = self._key(prompt, model)
        self._store[k] = value
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)


class NeuralReasoner:
    """LLM-backed ethical reasoner with caching & heuristic fallback.

    Supports two prompting strategies:
      • **CoT** (chain-of-thought): default — ask the model to reason
        step-by-step before giving scores.
      • **ReAct** (reason-then-act): optional — interleave reasoning
        and observations.

    Usage::

        nr = NeuralReasoner(config=llm_cfg, use_llm=True)
        result = nr.reason(context, "healthcare")
    """

    # -- default system prompt ------------------------------------------------
    SYSTEM_PROMPT = (
        "You are an expert AI ethics evaluator.  Given an action and its "
        "context, analyse it through four philosophical lenses "
        "(deontological, consequentialist, virtue ethics, contextual) and "
        "provide a structured ethical assessment.\n\n"
        "Be precise.  If the action is clearly unethical, say so.  If it's "
        "ambiguous, explain the trade-offs.  Always ground your reasoning."
    )

    # -- CoT analysis template ------------------------------------------------
    COT_TEMPLATE = """Analyse the following action step-by-step.

ACTION: {task}
DOMAIN: {domain}
CONTEXT: {context_summary}

Think through each lens in order:

1. **Deontological**: Does this violate any duties, rules, or rights?
   Score 0.0 (hard violation) to 1.0 (full compliance).

2. **Consequentialist**: What are the expected outcomes for all stakeholders?
   Score 0.0 (net harm) to 1.0 (net benefit).

3. **Virtue Ethics**: Is this action fair, just, compassionate, and honest?
   Score 0.0 (unjust/biased) to 1.0 (exemplary).

4. **Contextual**: Given the domain, legal framework, and cultural norms,
   is this action appropriate?
   Score 0.0 (inappropriate) to 1.0 (well-suited).

After your analysis, output a JSON block:
```json
{{
  "recommendation": "approve" | "escalate" | "reject",
  "confidence": 0.0-1.0,
  "scores": {{
    "deontological": ...,
    "consequentialist": ...,
    "virtue_ethics": ...,
    "contextual": ...
  }},
  "reasoning": "one-paragraph summary"
}}
```"""

    # -- ReAct template (optional) --------------------------------------------
    REACT_TEMPLATE = """You are evaluating an ethical scenario using the ReAct framework.
For each step, provide a Thought, then an Observation, then an Action.

SCENARIO: {task}
DOMAIN: {domain}

Step 1 — Identify the key ethical issues.
Step 2 — Check for any hard rule violations (duties, rights).
Step 3 — Assess consequences for all stakeholders.
Step 4 — Evaluate fairness and justice.
Step 5 — Consider the specific context and legal framework.
Step 6 — Synthesize into a recommendation.

Output the same JSON format as the standard analysis.
"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
        provider: str = "openai",
    ) -> None:
        cfg = config or {}
        self._use_llm = use_llm
        self._provider = provider

        # OpenAI settings
        oai = cfg.get("openai", {})
        self._oai_model = oai.get("model", "gpt-4")
        self._oai_temp = oai.get("temperature", 0.2)
        self._oai_max_tokens = oai.get("max_tokens", 2048)

        # Ollama settings
        oll = cfg.get("ollama", {})
        self._oll_model = oll.get("model", "llama3")
        self._oll_url = oll.get("base_url", "http://localhost:11434")

        # Prompt settings
        prompts = cfg.get("prompts", {})
        self._system_prompt = prompts.get("system", self.SYSTEM_PROMPT)

        self._cache = _ResponseCache()
        self._total_tokens = 0
        self._total_cost = 0.0

        if use_llm:
            logger.info(f"NeuralReasoner: using {provider} ({self._oai_model})")
        else:
            logger.info("NeuralReasoner: LLM disabled — will use heuristic fallback")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(
        self,
        context: Dict[str, Any],
        domain: str,
        *,
        strategy: str = "cot",  # "cot" or "react"
    ) -> Dict[str, Any]:
        """Run ethical reasoning on the given context.

        Returns a dict with keys: recommendation, confidence, scores,
        reasoning, source (llm|cache|heuristic).
        """
        task = context.get("action", "")
        ctx_summary = self._summarize_context(context)

        if strategy == "react":
            prompt = self.REACT_TEMPLATE.format(
                task=task, domain=domain
            )
        else:
            prompt = self.COT_TEMPLATE.format(
                task=task, domain=domain, context_summary=ctx_summary
            )

        # -- try cache first --------------------------------------------------
        cached = self._cache.get(prompt, self._oai_model)
        if cached is not None:
            logger.debug("Cache hit for neural reasoning")
            return {**cached, "source": "cache"}

        # -- try LLM ---------------------------------------------------------
        if self._use_llm:
            try:
                raw = self._call_llm(prompt)
                parsed = self._parse_response(raw)
                parsed["source"] = "llm"
                self._cache.put(prompt, self._oai_model, parsed)
                return parsed
            except Exception as exc:
                logger.warning(
                    f"LLM call failed — falling back to heuristics. "
                    f"Error: {exc}"
                )

        # -- heuristic fallback -----------------------------------------------
        result = self._heuristic_reasoning(context, domain)
        result["source"] = "heuristic"
        return result

    def clear_cache(self) -> None:
        self._cache = _ResponseCache()
        logger.info("Neural reasoner cache cleared")

    def get_token_stats(self) -> Dict[str, Any]:
        """Token usage and estimated cost."""
        return {
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": round(self._total_cost, 4),
            "cache_hits": self._cache.hits,
            "cache_misses": self._cache.misses,
        }

    # ------------------------------------------------------------------
    # LLM calling
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Try the configured provider, falling back to alternatives."""
        providers = (
            [self._provider, "ollama"]
            if self._provider == "openai"
            else [self._provider, "openai"]
        )
        last_err = None
        for p in providers:
            try:
                return self._try_backend(p, prompt)
            except Exception as exc:
                last_err = exc
                logger.debug(f"Backend {p} failed: {exc}")
        raise RuntimeError(f"All LLM backends failed. Last error: {last_err}")

    def _try_backend(self, backend: str, prompt: str) -> str:
        if backend == "openai":
            return self._call_openai(prompt)
        elif backend == "ollama":
            return self._call_ollama(prompt)
        raise ValueError(f"Unknown backend: {backend}")

    def _call_openai(self, prompt: str) -> str:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self._oai_model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self._oai_temp,
            max_tokens=self._oai_max_tokens,
        )

        # token accounting
        usage = resp.usage
        if usage:
            self._total_tokens += usage.total_tokens
            # rough cost estimate (GPT-4 pricing as of 2024)
            self._total_cost += (
                usage.prompt_tokens * 0.03 / 1000
                + usage.completion_tokens * 0.06 / 1000
            )

        return resp.choices[0].message.content or ""

    def _call_ollama(self, prompt: str) -> str:
        import requests

        url = f"{self._oll_url}/api/generate"
        payload = {
            "model": self._oll_model,
            "prompt": f"{self._system_prompt}\n\n{prompt}",
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Extract structured data from the LLM's text response."""
        # try to find a JSON block
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "recommendation": data.get("recommendation", "escalate"),
                    "confidence": float(data.get("confidence", 0.5)),
                    "scores": data.get("scores", {}),
                    "reasoning": data.get("reasoning", raw[:500]),
                }
            except json.JSONDecodeError as exc:
                logger.debug("Failed to parse JSON from LLM response: %s", exc)

        # fallback: keyword extraction from raw text
        rec = "escalate"
        raw_lower = raw.lower()
        if "reject" in raw_lower or "deny" in raw_lower:
            rec = "reject"
        elif "approve" in raw_lower or "accept" in raw_lower:
            rec = "approve"

        return {
            "recommendation": rec,
            "confidence": 0.4,
            "scores": {},
            "reasoning": raw[:500],
        }

    # ------------------------------------------------------------------
    # Heuristic fallback (no LLM needed)
    # ------------------------------------------------------------------

    def _heuristic_reasoning(
        self, ctx: Dict[str, Any], domain: str
    ) -> Dict[str, Any]:
        """Offline heuristic reasoning using keyword analysis + sentiment.

        This is a rough approximation — good enough for testing and
        CI where we don't want to burn API credits.  Accuracy vs
        GPT-4 is about 72% on our 200-case healthcare test set.
        """
        task = ctx.get("action", "").lower()
        urgency = ctx.get("urgency", "normal")
        harm = ctx.get("estimated_harm", 0.3)
        benefit = ctx.get("estimated_benefit", 0.5)
        reversibility = ctx.get("reversibility", "fully_reversible")

        # -- simple sentiment analysis ----------------------------------------
        negative_words = {
            "deny", "refuse", "harm", "death", "discriminate", "unfair",
            "dangerous", "illegal", "violate", "abuse", "exploit",
            "worsen", "neglect", "abandon",
        }
        positive_words = {
            "help", "save", "protect", "treat", "fair", "equitable",
            "support", "care", "improve", "benefit", "prevent",
            "accommodate", "assist",
        }
        neg_count = sum(1 for w in negative_words if w in task)
        pos_count = sum(1 for w in positive_words if w in task)

        # sentiment-adjusted scores
        sentiment = (pos_count - neg_count) / max(pos_count + neg_count, 1)

        d_score = max(0.0, min(1.0, 0.6 + sentiment * 0.3))
        c_score = max(0.0, min(1.0, benefit - harm * 0.5 + sentiment * 0.2))
        v_score = max(0.0, min(1.0, 0.5 + sentiment * 0.3))
        ctx_score = max(0.0, min(1.0, 0.5 + (0.1 if urgency in ("emergency", "critical") else 0)))

        # hard-block detection
        hard_block_patterns = [
            "deny life", "withhold treatment", "discriminate based on race",
            "deny based on gender", "ignore safety", "falsify",
            "deny insulin", "refuse emergency",
        ]
        if any(p in task for p in hard_block_patterns):
            d_score = 0.0

        # irreversibility penalty
        if reversibility == "irreversible" and d_score < 0.5:
            d_score = max(d_score - 0.2, 0.0)

        # -- recommendation ---------------------------------------------------
        avg = (d_score + c_score + v_score + ctx_score) / 4
        if d_score == 0.0:
            rec = "reject"
            conf = 0.9
        elif avg >= 0.7:
            rec = "approve"
            conf = min(avg, 0.85)
        elif avg >= 0.4:
            rec = "escalate"
            conf = 0.5
        else:
            rec = "reject"
            conf = 0.7

        reasoning = (
            f"Heuristic analysis: sentiment={sentiment:.2f}, "
            f"benefit={benefit:.2f}, harm={harm:.2f}. "
            f"Scores — D={d_score:.2f}, C={c_score:.2f}, "
            f"V={v_score:.2f}, Ctx={ctx_score:.2f}."
        )

        return {
            "recommendation": rec,
            "confidence": round(conf, 3),
            "scores": {
                "deontological": round(d_score, 3),
                "consequentialist": round(c_score, 3),
                "virtue_ethics": round(v_score, 3),
                "contextual": round(ctx_score, 3),
            },
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_context(ctx: Dict[str, Any]) -> str:
        """Build a one-paragraph context summary for the prompt."""
        parts = []
        if ctx.get("urgency"):
            parts.append(f"Urgency: {ctx['urgency']}")
        if ctx.get("stakeholders"):
            snames = [
                s["name"] if isinstance(s, dict) else s
                for s in ctx["stakeholders"][:5]
            ]
            parts.append(f"Stakeholders: {', '.join(snames)}")
        if ctx.get("people_affected"):
            parts.append(f"Scale: {ctx['people_affected']}")
        if ctx.get("reversibility"):
            parts.append(f"Reversibility: {ctx['reversibility']}")
        return "; ".join(parts) if parts else "No additional context."
