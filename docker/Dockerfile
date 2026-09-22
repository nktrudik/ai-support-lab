FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev
COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts ./scripts
RUN useradd --create-home app && mkdir -p artifacts data && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "ai_support_lab.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
