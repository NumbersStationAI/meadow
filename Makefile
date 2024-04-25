.PHONY: install test

build:
	poetry build --format wheel -vvv

install:
	poetry install
	poetry run pre-commit install

format:
	poetry run ruff check --fix meadow/ tests/
	poetry run ruff format

check:
	poetry run ruff check meadow/
	poetry run ruff check tests/
	poetry run ruff format
	poetry run mypy .

test:
	poetry run pytest