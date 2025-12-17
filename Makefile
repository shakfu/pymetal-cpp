PROJECT_NAME = pymetal
VERSION = 0.1.0

.PHONY: all build wheel clean test snap lint lint-fix format format-check typecheck check publish-test publish

all: build


build:
	@uv sync --reinstall-package pymetal

wheel:
	@uv build

install:
	@uv pip uninstall pymetal || true
	@uv pip install dist/pymetal-*.whl

clean:
	rm -rf build dist src/pymetal/*.so

test:
	@.venv/bin/pytest tests/ -v

repl:
	@uv run python -m pychuck tui

snap:
	@git add --all . && git commit -m 'snap' && git push

# Linting and formatting (ruff)
lint:
	@.venv/bin/ruff check src/ tests/

lint-fix:
	@.venv/bin/ruff check --fix src/ tests/

format:
	@.venv/bin/ruff format src/ tests/

format-check:
	@.venv/bin/ruff format --check src/ tests/

# Type checking (mypy)
typecheck:
	@.venv/bin/mypy src/

# Package validation and publishing (twine)
check:
	@.venv/bin/twine check dist/*

publish-test:
	@.venv/bin/twine upload --repository testpypi dist/*

publish:
	@.venv/bin/twine upload dist/*
