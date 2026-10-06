# Shortcuts for macOS and Linux. On Windows, run the `ledgerlens` commands directly.
.PHONY: setup test lint format typecheck all demo serve clean
BIN := .venv/bin
setup:
	python3 -m venv .venv && $(BIN)/pip install --upgrade pip && $(BIN)/pip install -e ".[dev]"
test:
	$(BIN)/pytest -q
lint:
	$(BIN)/ruff check src tests && $(BIN)/ruff format --check src tests
format:
	$(BIN)/ruff check --fix src tests && $(BIN)/ruff format src tests
typecheck:
	$(BIN)/mypy src
all:
	$(BIN)/ledgerlens all
demo:
	$(BIN)/ledgerlens all --provider simulated
serve:
	$(BIN)/ledgerlens serve
clean:
	rm -rf output/data output/*.csv output/*.json output/*.jsonl output/*.html
