.PHONY: setup lint format test smoke smoke-eval docker-up docker-down docker-smoke docker-ml-smoke \
	sanity-api sanity

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
	uv run --extra ml --extra plots python -m voxguard.evaluate --config configs/eval_smoke.yaml

docker-up:
	docker compose up --build --detach --wait

docker-down:
	docker compose down

docker-smoke:
	VOXGUARD_DOCKER_TESTS=1 uv run pytest tests/test_docker_smoke.py -v

# INSTALL_ML=1 image: CPU-only torch audit + real detector on an untrained checkpoint
docker-ml-smoke:
	VOXGUARD_DOCKER_ML_TESTS=1 uv run pytest tests/test_docker_ml.py -v

# Real-audio sanity check (docs/real-audio-testing.md). Diagnostic only; reports stay private.
# 1) a scratch API on a NEW database outside the repo (never the demo database), e.g.
#    VOXGUARD_MODEL_REPO=voxguard-team/... make sanity-api SANITY_DB=$$HOME/private/sanity-1.db
sanity-api:
	uv run --extra api --extra ml python scripts/real_audio_sanity.py serve --db "$(SANITY_DB)" $(SANITY_API_ARGS)

# 2) in another terminal, score the labelled, consented files through its public API
#    make sanity SANITY_URL=http://127.0.0.1:8100 LABELS=… AUDIO=… OUT=…
sanity:
	uv run python scripts/real_audio_sanity.py run --api-url "$(SANITY_URL)" --labels "$(LABELS)" \
		--audio-dir "$(AUDIO)" --out "$(OUT)" $(SANITY_ARGS)
