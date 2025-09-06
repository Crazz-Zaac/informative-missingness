FROM python:3.11.9-slim-bookworm AS base

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /project

# Install system dependencies with security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc6-dev \
        libgomp1 \
        curl \
        ca-certificates && \
    # Clean up package cache and temporary files
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    # Remove setuid/setgid bits from potentially dangerous binaries
    find /usr/bin /usr/sbin /bin /sbin -perm /6000 -type f -exec chmod a-s {} \; || true

# Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt .
# Copy wheels and requirements
COPY --chown=appuser:appuser wheels/ ./wheels/
COPY --chown=appuser:appuser requirements.txt .

# Install using local wheels only
RUN pip install --no-index --find-links=./wheels -r requirements.txt


# Switch to non-root user for pip install
USER appuser

# Copy project files
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PYTHONPATH=/project
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/appuser/.local/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Use non-root user
USER appuser

# Default command (can be overridden in docker-compose)
CMD ["python", "run_exp.py"]