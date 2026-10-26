# The reviewer service (FastAPI + agents + indexing).
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY benchmark ./benchmark
RUN pip install --no-cache-dir .

RUN useradd --create-home app && mkdir -p /data && chown app /data
# The app user must be able to reach the Docker socket to launch sandboxes; compose adds the
# socket's group via `group_add`.
USER app
# Pre-download Chroma's default embedding model at build time (into the app user's cache) so the
# first review doesn't stall on a download.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warmup'])"
ENV CHROMA_PATH=/data/chroma WORKSPACE_DIR=/data/workspaces

EXPOSE 8000
CMD ["uvicorn", "reviewer.main:app", "--host", "0.0.0.0", "--port", "8000"]
