# LivingBench Docker Image
# Multi-stage build for optimal image size

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install build && \
    pip install -e ".[groq]"

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY livingbench/ ./livingbench/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY pyproject.toml .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create output directory
RUN mkdir -p /app/outputs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-m", "experiments.run_experiment", "--help"]

# ============================================
# Labels
# ============================================
LABEL org.opencontainers.image.title="LivingBench"
LABEL org.opencontainers.image.description="LLM Evaluation Framework"
LABEL org.opencontainers.image.version="0.1.0"
