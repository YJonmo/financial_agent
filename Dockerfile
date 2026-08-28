FROM python:3.12-slim

# uv (dependency manager this project uses) - copied in as a static binary,
# per https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first, in their own layer, so code-only changes below
# don't invalidate the dependency cache.
COPY pyproject.toml ./
RUN uv sync --no-install-project

# Now add the actual project (source + dataset) and install it too, which
# registers the `main` console script defined in pyproject.toml.
COPY README.md ./README.md 
COPY src/ ./src/
COPY data/ ./data/
RUN uv sync

# `evaluate` writes result JSON files here - mount a volume at this path to
# persist them on the host (see README/usage notes for the docker run command).
RUN mkdir -p eval_results

ENV PYTHONUNBUFFERED=1

# API keys (OPENAI_API_KEY / GROQ_API_KEY) are intentionally NOT baked into
# the image - pass them at `docker run` time with --env-file .env or -e.
ENTRYPOINT ["uv", "run", "main"]
CMD ["--help"]
