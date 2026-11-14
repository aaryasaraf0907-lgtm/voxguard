.PHONY: setup lint format test smoke smoke-eval docker-up docker-down docker-smoke docker-ml-smoke

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

docker-up:
	docker compose up --build --detach --wait

docker-down:
	docker compose down

docker-smoke:
	VOXGUARD_DOCKER_TESTS=1 uv run pytest tests/test_docker_smoke.py -v

# INSTALL_ML=1 image: CPU-only torch audit + real detector on an untrained checkpoint
docker-ml-smoke:
	VOXGUARD_DOCKER_ML_TESTS=1 uv run pytest tests/test_docker_ml.py -v
