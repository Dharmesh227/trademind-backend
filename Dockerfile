# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install \
       fastapi uvicorn[standard] sqlalchemy aiosqlite pydantic \
       pydantic-settings pandas numpy scikit-learn httpx \
       apscheduler python-dotenv loguru aiofiles alembic

# ── Stage 2: Runtime ───────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="TradeMind AI Team"
LABEL description="TradeMind AI — Self-Improving NSE Trading Intelligence"

RUN groupadd -r trademind && useradd -r -g trademind -d /app trademind

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

RUN mkdir -p /app/models /app/data /app/logs \
    && chown -R trademind:trademind /app

USER trademind

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); assert r.status_code==200"

CMD ["uvicorn", "trademind.main:app", "--host", "0.0.0.0", "--port", "8000"]
