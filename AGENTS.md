# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/` (CLI `src/cli.py`, REST API `src/api.py`, DI v1 API `src/api_v1_di.py`)
- Models/Pipeline: `src/modeling/`, `src/pipeline/`, exporters `src/exporters/`
- Configs & Assets: `configs/`, sample data `data/samples/`, results `results/`, weights `weights/`
- Ops: `scripts/`, `infra/`, server `server/`, docs `docs/requirements_definition/`
- Add tests under `tests/` (e.g., `tests/test_pipeline.py`).

## Build, Test, and Development Commands
- `make setup-local` then `source venv/bin/activate` to install deps and scaffold folders.
- Run API (dev): `make serve-dev` → FastAPI at `http://localhost:8001`.
- CLI example: `docja path/to/input.pdf -f md -o results/ --layout --table --reading-order`.
- Tests: `make test` (pytest + coverage). Models/pipeline smoke: `make test-models`, `make test-pipeline`.
- Lint/Format: `make lint` (black, isort, flake8, mypy), `make format`.
- Eval/Serving: `bash scripts/run_eval.sh`, `make serve` (vLLM), Docker: `make docker-build`/`make docker-run`.

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indent, type hints required; docstrings for public APIs.
- File names: `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Keep Clean Architecture boundaries: presentation (`src/cli.py`/API), application (orchestration), domain (core logic), infrastructure (model I/O).
- Do not copy third‑party code or ship model weights; keep secrets in `.env` only.

## Testing Guidelines
- Framework: `pytest`; target coverage ≥ 80% for new/changed code.
- Place tests in `tests/` with `test_*.py`; prefer small fixtures under `data/samples/`.
- Add fast unit tests for `src/pipeline/*` and contract tests for APIs (Pydantic models).

## Commit & Pull Request Guidelines
- Commit style: concise imperative subject; reference issues when applicable.
- Pre‑PR checklist: run `make format && make lint && make test` and include:
  - What/Why, affected modules, perf impact (p50/p95 if relevant), and CLI/API examples.
- Use `make commit M="message"` to push; prefer PRs that are scoped and reviewable (<500 LOC).

## Security & Configuration Tips
- Required env: `DOCJA_API_KEY`, `DOCJA_LLM_PROVIDER`, `DOCJA_LLM_ENDPOINT`, `AWS_PROFILE`, `KEY_NAME`, `S3_BUCKET`.
- Do not commit secrets, large artifacts, or licensed datasets. Verify dataset licenses per `docs/requirements_definition/08_compliance_requirements.md`.

## Architecture Overview
- Modular OCR (det/rec/layout/table) + reading order; Unified Doc JSON is the single source of truth; LLM via Ollama/vLLM. See `docs/requirements_definition/` for detailed requirements.

