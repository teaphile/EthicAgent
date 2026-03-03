"""External benchmark adapters.

These adapters convert published ethical AI datasets into
EthicAgent's ScenarioCase format for direct evaluation.

Currently supported:
  - ETHICS dataset (moral reasoning)
  - BBQ (bias in question answering)

NOTE: datasets are downloaded on first use and cached locally.
We intentionally keep the download optional so CI doesn't
need internet access — tests use pre-cached samples.
"""

from __future__ import annotations

from ethicagent.benchmarks.external.ethics_adapter import EthicsDatasetAdapter
from ethicagent.benchmarks.external.bbq_adapter import BBQAdapter

__all__ = [
    "EthicsDatasetAdapter",
    "BBQAdapter",
]
