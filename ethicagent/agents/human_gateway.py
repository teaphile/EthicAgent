"""Human Gateway — human-in-the-loop review interface.

When the pipeline verdict is ESCALATE, the decision lands here.
Supports two modes:

  1. Synchronous callback — caller provides a function that returns
     a human verdict immediately (good for CLI tools).
  2. Async queue — the decision is queued and can be resolved later
     via ``resolve_review()`` (good for web UIs).

# NOTE: in production this would integrate with Slack / email / a
#       custom dashboard.  For now it's in-process.
# TODO: add timeout-based auto-reject (configurable per domain).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMED_OUT = "timed_out"


@dataclass
class ReviewRequest:
    """A queued human review request."""
    review_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    decision: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    priority: str = "medium"
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: Optional[str] = None
    reviewer: Optional[str] = None
    reviewer_notes: str = ""


class HumanGateway:
    """Interface for human-in-the-loop ethical review."""

    def __init__(
        self,
        review_callback: Optional[Callable] = None,
        auto_reject_timeout_s: float = 3600.0,
    ) -> None:
        self._callback = review_callback
        self._queue: Dict[str, ReviewRequest] = {}
        self._resolved: List[ReviewRequest] = []
        self._timeout = auto_reject_timeout_s

    def escalate(
        self,
        decision: Any,
        context: Dict[str, Any],
        priority: str = "medium",
    ) -> Dict[str, Any]:
        """Escalate a decision for human review.

        If a callback is registered, process synchronously.
        Otherwise queue for later resolution.
        """
        req = ReviewRequest(
            decision=decision, context=context, priority=priority
        )

        if self._callback:
            return self._process_with_callback(req)

        return self._queue_for_review(req)

    def resolve_review(
        self,
        review_id: str,
        status: str,
        reviewer: str = "anonymous",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Resolve a pending review."""
        req = self._queue.get(review_id)
        if not req:
            return {"error": f"Review {review_id} not found"}

        req.status = ReviewStatus(status)
        req.resolved_at = datetime.now(timezone.utc).isoformat()
        req.reviewer = reviewer
        req.reviewer_notes = notes

        del self._queue[review_id]
        self._resolved.append(req)

        logger.info(f"Review {review_id} resolved: {status} by {reviewer}")
        return {
            "review_id": review_id,
            "status": status,
            "reviewer": reviewer,
        }

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        return [
            {
                "review_id": r.review_id,
                "priority": r.priority,
                "status": r.status.value,
                "created_at": r.created_at,
            }
            for r in self._queue.values()
        ]

    def get_review_statistics(self) -> Dict[str, Any]:
        return {
            "pending": len(self._queue),
            "resolved": len(self._resolved),
            "total": len(self._queue) + len(self._resolved),
        }

    # -- internal -------------------------------------------------------------

    def _process_with_callback(self, req: ReviewRequest) -> Dict[str, Any]:
        try:
            result = self._callback(req.decision, req.context)
            req.status = ReviewStatus.APPROVED if result.get("approved") else ReviewStatus.REJECTED
            req.resolved_at = datetime.now(timezone.utc).isoformat()
            self._resolved.append(req)
            return {
                "review_id": req.review_id,
                "status": req.status.value,
                "callback_result": result,
            }
        except Exception as exc:
            logger.error(f"Review callback failed: {exc}")
            return self._queue_for_review(req)

    def _queue_for_review(self, req: ReviewRequest) -> Dict[str, Any]:
        self._queue[req.review_id] = req
        logger.info(
            f"Decision queued for human review: {req.review_id} "
            f"(priority={req.priority})"
        )
        return {
            "review_id": req.review_id,
            "status": "pending",
            "message": "Decision has been queued for human review.",
        }
