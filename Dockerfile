FROM python:3.11-slim AS base

LABEL maintainer="radhahai"
LABEL description="EthicAgent — Neuro-Symbolic Ethical Reasoning"

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py requirements.txt README.md ./
COPY ethicagent/ ./ethicagent/
RUN pip install --no-cache-dir . && \
    apt-get purge -y --auto-remove gcc || true

COPY . .

# ── test stage ──────────────────────────────────────────────
FROM base AS test
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
CMD ["pytest", "tests/", "--tb=short", "-q"]

# ── dashboard stage ─────────────────────────────────────────
FROM base AS dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ── default ─────────────────────────────────────────────────
FROM base AS production
CMD ["python", "-m", "ethicagent"]
