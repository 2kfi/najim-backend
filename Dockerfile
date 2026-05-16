# =============================================================================
# Najim Backend — Dockerfile
# Multi-stage build for minimal size.
# =============================================================================

# ─── Stage 1: Builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /build
COPY requirements.txt .
RUN pip install --user --upgrade pip && \
    pip install --user --prefer-binary -r requirements.txt

# ─── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Runtime system deps (audio codecs, piper needs libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy app code
COPY core/ core/
COPY sessions/ sessions/
COPY tools/ tools/
COPY pipeline/ pipeline/
COPY api/ api/
COPY app.py .
COPY config.yaml .

# ─── Expose ────────────────────────────────────────────────────────────────
EXPOSE 8080

# ─── Entrypoint ────────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
