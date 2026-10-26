# Base image for the test sandbox. Kept minimal: Python + pytest. The repo's own dependencies are
# installed into a derived, cached image at review time (see reviewer/sandbox/runner.py).
FROM python:3.11-slim

RUN pip install --no-cache-dir pytest==8.3.3 pytest-timeout==2.3.1 \
 && useradd --uid 1000 --create-home sandbox \
 && mkdir -p /work/repo /work/out \
 && chown -R sandbox:sandbox /work

ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
USER sandbox
WORKDIR /work/repo
