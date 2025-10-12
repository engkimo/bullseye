# Repository Guidelines

This document summarizes how to contribute effectively to this repository.

## Project Structure & Module Organization
- Source: `src/` (CLI `src/cli.py`, REST `src/api.py`, DI v1 `src/api_v1_di.py`)
- Models/Pipeline: `src/modeling/`, `src/pipeline/`, exporters `src/exporters/`
- Configs & Assets: `configs/`, samples `data/samples/`, results `results/`, weights `weights/`
- Ops: `scripts/`, infra `infra/`, server `server/`, docs `docs/requirements_definition/`
- Tests: `tests/` (e.g., `tests/test_pipeline.py`)

## Build, Test, and Development Commands
- Setup: `make setup-local && source venv/bin/activate` (create venv and deps)
- API (dev): `make serve-dev` (FastAPI at http://localhost:8001)
- CLI: `docja path/to/input.pdf -f md -o results/ --layout --table --reading-order`
- Tests: `make test` (all), `make test-models`, `make test-pipeline`
- Lint/Format: `make lint` (black/isort/flake8/mypy), `make format`
- Serving/Eval: `make serve` (vLLM), `bash scripts/run_eval.sh`; Docker: `make docker-build` / `make docker-run`

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indent, type hints for public APIs; concise docstrings.
- Naming: files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Architecture: follow Clean Architecture boundaries (presentation, application, domain, infrastructure). Keep modules cohesive; avoid cross‑layer imports; no business logic in CLI/API.

## Testing Guidelines
- Framework: `pytest`; coverage ≥ 80%.
- Layout: tests under `tests/test_*.py`; prefer tiny fixtures from `data/samples/`.
- Focus: fast unit tests for `src/pipeline/*`; contract tests for Pydantic models and REST.
- Run: `make test` or target-specific commands above.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (≤72 chars); one logical change per commit.
- Before PR: `make format && make lint && make test`; include What/Why, affected modules, performance impact (p50/p95 if relevant), and CLI/API examples.
- PRs: link issues, attach screenshots/logs for UI/CLI results, keep <500 LOC when possible.

## Security & Configuration Tips
- Required env: `DOCJA_API_KEY`, `DOCJA_LLM_PROVIDER`, `DOCJA_LLM_ENDPOINT`, `AWS_PROFILE`, `KEY_NAME`, `S3_BUCKET`.
- Never commit secrets, large artifacts, or licensed datasets; verify licenses (see `docs/requirements_definition/08_compliance_requirements.md`). Use `.env` locally.

## Architecture Overview (for Contributors)
- Modular OCR (det/rec/layout/table) + reading order; Unified Doc JSON is the single source of truth.
- LLM via Ollama/vLLM; DI v1 endpoints under FastAPI. When adding modules, respect boundaries, update configs/tests/docs, and include a CLI/API usage example.
