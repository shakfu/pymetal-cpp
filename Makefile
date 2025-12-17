PROJECT_NAME = pymetal
VERSION = 0.1.0

.PHONY: all build wheel clean test snap lint lint-fix format format-check \
		typecheck check publish-test publish release

all: build


build:
	@uv sync --reinstall-package pymetal-cpp

wheel:
	@uv build

release:
	@uv build --sdist
	@uv build --wheel --python 3.9
	@uv build --wheel --python 3.10
	@uv build --wheel --python 3.11
	@uv build --wheel --python 3.12

clean:
	rm -rf build dist src/pymetal/*.so

test:
	@uv run pytest tests/ -v

snap:
	@git add --all . && git commit -m 'snap' && git push

# Linting and formatting (ruff)
lint:
	@uv run ruff check src/ tests/

lint-fix:
	@uv run ruff check --fix src/ tests/

format:
	@uv run ruff format src/ tests/

format-check:
	@uv run ruff format --check src/ tests/

# Type checking (mypy)
typecheck:
	@.venv/bin/mypy src/

# Package validation and publishing (twine)
check:
	@uv run twine check dist/*

publish-test:
	@uv run twine upload --repository testpypi dist/*

publish:
	@uv run twine upload dist/*
