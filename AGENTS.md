# Repository Guidelines

This guide helps contributors ship high‑quality changes quickly.

## Project Structure & Module Organization
- Source: `src/` (CLI `src/cli.py`, REST `src/api.py`, DI v1 `src/api_v1_di.py`)
- Models/Pipeline: `src/modeling/`, `src/pipeline/`, exporters `src/exporters/`
- Configs & Assets: `configs/`; samples `data/samples/`; results `results/`; weights `weights/`
- Ops: `scripts/`, `infra/`, `server/`; docs: `docs/requirements_definition/`
- Tests: `tests/` (e.g., `tests/test_pipeline.py`)

## Build, Test, and Development Commands
- Setup: `make setup-local && source venv/bin/activate` (create venv + deps)
- Run API (dev): `make serve-dev` (FastAPI → http://localhost:8001)
- CLI (example): `docja input.pdf -f md -o results/ --layout --table --reading-order`
- Tests: `make test` (all) | `make test-models` | `make test-pipeline`
- Lint/Format: `make lint` (black/isort/flake8/mypy) | `make format`
- Serving/Eval: `make serve` (vLLM) | `bash scripts/run_eval.sh` | Docker: `make docker-build` / `make docker-run`

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indent, type hints; docstrings for public APIs.
- Names: files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Architecture: Clean Architecture boundaries（presentation / application / domain / infrastructure）。No cross‑layer imports; avoid business logic in CLI/API.

## Testing Guidelines
- Framework: `pytest`; coverage ≥ 80%.
- Location: `tests/test_*.py`; use small fixtures from `data/samples/`.
- Focus: fast unit tests for `src/pipeline/*`; contract tests for Pydantic models + REST.
- Run locally: `make test` or target-specific commands above.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (≤72 chars); one logical change per commit.
- Before PR: `make format && make lint && make test`; include What/Why, affected modules, perf impact（p50/p95 if relevant）, and CLI/API examples.
- PRs: link issues; add screenshots/logs for UI/CLI; target <500 LOC when feasible.

## Security & Configuration Tips
- Required env: `DOCJA_API_KEY`, `DOCJA_LLM_PROVIDER`, `DOCJA_LLM_ENDPOINT`, `AWS_PROFILE`, `KEY_NAME`, `S3_BUCKET`。
- Do not commit secrets, large artifacts, or licensed datasets; keep credentials in `.env`. Confirm licenses（see `docs/requirements_definition/08_compliance_requirements.md`）。

## Architecture Overview（for contributors）
- Modular OCR（det/rec/layout/table）+ reading order; Unified Doc JSON is the single source of truth.
- LLM via Ollama/vLLM; DI v1 endpoints under FastAPI. When adding modules, respect boundaries, update configs/tests/docs, and provide a CLI/API example.
