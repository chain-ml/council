.PHONY: format lint dev-lint

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

format:
	black .

dev-lint:
	mypy .
	black .
	ruff check . --fix

lint:
	mypy .
	black . --check
	ruff check .


test:
	pytest tests

unit-test:
	pytest tests/unit

integration-test:
	pytest tests/integration

notebook-test:
	pytest tests/notebooks
