# Shortcuts for macOS and Linux. On Windows, run the `assurance-lab` commands directly.
.PHONY: setup test lint format typecheck check train baseline hardened report all demo serve docker clean

PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	$(BIN)/pre-commit install || true

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

typecheck:
	$(BIN)/mypy src

check: lint typecheck test

train:
	$(BIN)/assurance-lab train

baseline:
	$(BIN)/assurance-lab redteam --config configs/baseline.yaml

hardened:
	$(BIN)/assurance-lab redteam --config configs/hardened.yaml

report:
	$(BIN)/assurance-lab report

all:
	$(BIN)/assurance-lab all

demo:
	$(BIN)/assurance-lab all --provider simulated

serve:
	$(BIN)/assurance-lab serve

docker:
	docker compose up --build

clean:
	rm -rf reports/runs reports/*.html reports/*.md reports/*.json models/*.joblib data/generated
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
