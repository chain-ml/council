.PHONY: format lint dev-lint

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

format:
	black .
	isort .

dev-lint:
	mypy .
	black .
	ruff check . --fix
	isort .

lint:
	mypy .
	black . --check
	ruff check .
	pylint council/. --max-line-length 120 --disable=R,C,I,W1203,W0107 --fail-under=9
	isort . --check-only

test:
	pytest tests

unit-test:
	pytest tests/unit

integration-test:
	pytest tests/integration

notebook-test:
	pytest tests/notebooks
