# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/` (CLI `src/cli.py`, REST `src/api.py`, DI v1 `src/api_v1_di.py`)
- Models/Pipeline: `src/modeling/`, `src/pipeline/`, exporters `src/exporters/`
- Configs & Assets: `configs/`, samples `data/samples/`, results `results/`, weights `weights/`
- Ops: `scripts/`, `infra/`, server `server/`, docs `docs/requirements_definition/`
- Tests: `tests/` (e.g., `tests/test_pipeline.py`)

## Build, Test, and Development Commands
- Setup: `make setup-local && source venv/bin/activate`
- Run API (dev): `make serve-dev` → FastAPI at `http://localhost:8001`
- CLI example: `docja path/to/input.pdf -f md -o results/ --layout --table --reading-order`
- Tests: `make test` (pytest+coverage), `make test-models`, `make test-pipeline`
- Lint/Format: `make lint` (black/isort/flake8/mypy), `make format`
- Serving/Eval: `make serve` (vLLM), `bash scripts/run_eval.sh`; Docker: `make docker-build`/`make docker-run`

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, type hints required; docstrings for public APIs
- Names: files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`
- Clean Architecture boundaries: presentation (`src/cli.py`/API), application (orchestration), domain (core), infrastructure (model I/O)
- Do not copy third‑party code or ship model weights; secrets only via `.env`

## Testing Guidelines
- Framework: `pytest`; new/changed code coverage ≥ 80%
- Tests live under `tests/test_*.py`; prefer small fixtures in `data/samples/`
- Add fast unit tests for `src/pipeline/*`; contract tests for APIs (Pydantic)
- Run locally: `make test` or focused targets above

## Commit & Pull Request Guidelines
- Commits: short imperative subject (≤72 chars), scope narrowly
- Pre‑PR: `make format && make lint && make test`; include What/Why, affected modules, perf impact (p50/p95 if relevant), CLI/API examples
- PRs: link issues, include screenshots/logs for UI/CLI, keep <500 LOC when possible

## Security & Configuration Tips
- Required env: `DOCJA_API_KEY`, `DOCJA_LLM_PROVIDER`, `DOCJA_LLM_ENDPOINT`, `AWS_PROFILE`, `KEY_NAME`, `S3_BUCKET`
- Never commit secrets, large artifacts, or licensed datasets; verify licenses per `docs/requirements_definition/08_compliance_requirements.md`

## Architecture Overview (for contributors)
- Modular OCR (det/rec/layout/table) + reading order; Unified Doc JSON is the single source of truth; LLM via Ollama/vLLM
- When adding modules, respect boundaries, update configs/tests/docs, and provide a CLI/API example
