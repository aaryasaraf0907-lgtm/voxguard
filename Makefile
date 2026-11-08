.PHONY: setup lint format test smoke smoke-eval

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

SYN := data/processed/synthetic
smoke-eval: smoke
	uv run --extra ml python -m voxguard.data.augment --manifest $(SYN)/manifests/synthetic_train.csv \
		--data-root $(SYN) --out-root $(SYN) --out-manifest $(SYN)/manifests/synthetic_train_aug.csv \
		--codecs g711_mulaw opus --n-variants 1
	uv run --extra ml python -m voxguard.data.augment --manifest $(SYN)/manifests/synthetic_dev.csv \
		--data-root $(SYN) --out-root $(SYN) --out-manifest $(SYN)/manifests/synthetic_dev_per_codec.csv \
		--codecs g711_mulaw opus --mode per-codec --no-clean
	uv run --extra ml python -m voxguard.train --config configs/rawnet2_smoke_aug.yaml --no-wandb
	uv run --extra ml python -m voxguard.evaluate --config configs/eval_smoke.yaml
