"""Tests for core audit logger module."""

from __future__ import annotations

import json
import os
import tempfile

from ethicagent.core.logger import AuditEntry, AuditLogger


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_creation(self):
        entry = AuditEntry(
            event_type="decision",
            message="Loan approved",
            data={"eds": 0.85},
        )
        assert entry.event_type == "decision"
        assert entry.message == "Loan approved"
        assert entry.timestamp is not None

    def test_to_dict(self):
        entry = AuditEntry(
            event_type="test",
            message="Test message",
        )
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["event_type"] == "test"
        assert "timestamp" in d


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_initialization(self):
        logger = AuditLogger()
        assert logger is not None

    def test_log_event(self):
        logger = AuditLogger()
        logger.log_event("test", "Test event", {"key": "value"})
        assert len(logger.entries) >= 1

    def test_log_decision(self):
        logger = AuditLogger()
        logger.log_decision(
            task="Approve loan",
            verdict="approve",
            eds_score=0.85,
            data={"domain": "finance"},
        )
        entries = [e for e in logger.entries if e.event_type == "decision"]
        assert len(entries) >= 1

    def test_log_conflict(self):
        logger = AuditLogger()
        logger.log_conflict(
            conflict_type="philosophy_disagreement",
            details={"between": ["deontological", "consequentialist"]},
        )
        entries = [e for e in logger.entries if e.event_type == "conflict"]
        assert len(entries) >= 1

    def test_log_error(self):
        logger = AuditLogger()
        logger.log_error("Test error", error=ValueError("test"))
        entries = [e for e in logger.entries if e.event_type == "error"]
        assert len(entries) >= 1

    def test_export_to_json(self):
        logger = AuditLogger()
        logger.log_event("test", "Export test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            logger.export_to_json(filepath)
            assert os.path.exists(filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) >= 1
        finally:
            os.unlink(filepath)

    def test_export_to_csv(self):
        logger = AuditLogger()
        logger.log_event("test", "CSV test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            logger.export_to_csv(filepath)
            assert os.path.exists(filepath)
            with open(filepath) as f:
                content = f.read()
            assert "test" in content
        finally:
            os.unlink(filepath)
