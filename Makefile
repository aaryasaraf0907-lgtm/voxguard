.PHONY: setup lint format test smoke

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

smoke:
	uv run --extra ml python -m voxguard.data.synthetic --n 100
	uv run --extra ml python -m voxguard.train --config configs/rawnet2_smoke.yaml --max-samples 100 --no-wandb
