# Dockerfile (docker/Dockerfile)
- This project uses a custom multi-stage Dockerfile located at docker/Dockerfile.

## Key points:
- `Base image`: Built on python:3.11.9-slim-bookworm.
- `Non-root user`: Creates and runs as appuser for improved security.

- `System dependencies`: Installs essential build tools (gcc, g++, libc6-dev, libgomp1) required for compiling certain Python packages.

- `Wheels installation`:
    - The Dockerfile is designed to install Python packages from pre-downloaded `.whl` files (inside the `wheels/` directory).
    - This avoids issues with downloading from PyPI on HPC clusters and ensures reproducible, dependency-locked builds.

- `Caching optimization`: Requirements and wheels are copied first, so Docker only rebuilds layers when dependencies change.
- `Security hardening`: Removes unnecessary package caches and strips setuid/setgid bits from potentially risky binaries.
- `Healthcheck`: Ensures the container is alive by running a lightweight Python check every 30s.
- `Default command`: Runs python run_exp.py, but this can be overridden in docker-compose.yml.

👉 In short: this Dockerfile ensures reproducible, secure, and portable environments — especially useful when building `.whl` files for running on HPC systems.