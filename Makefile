.PHONY: install run migrate seed test lint format typecheck generate-data train-sklearn train-torch train-lightning run-mcp
install:
	uv sync --locked --all-extras
run:
	uv run uvicorn ai_support_lab.main:create_app --factory --reload
migrate:
	uv run alembic upgrade head
seed:
	uv run python scripts/seed_db.py
test:
	uv run pytest -q
lint:
	uv run ruff check .
	uv run ruff format --check .
format:
	uv run ruff check . --fix
	uv run ruff format .
typecheck:
	uv run mypy
generate-data:
	uv run python scripts/generate_dataset.py
train-sklearn:
	uv run python scripts/train_sklearn.py
train-torch:
	uv run python scripts/train_torch.py
train-lightning:
	uv run --extra lightning python scripts/train_lightning.py
run-mcp:
	uv run python -m ai_support_lab.mcp.server
