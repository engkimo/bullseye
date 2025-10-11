.PHONY: help setup setup-local setup-aws train-all train-det train-rec train-layout train-table train-lora eval-all serve ssh infra-plan infra-apply infra-destroy clean test lint

# Default target
help:
	@echo "DocJA - Japanese Document AI System"
	@echo ""
	@echo "Setup commands:"
	@echo "  make setup-local    - Setup local development environment"
	@echo "  make setup-uv       - Setup local env using uv (.venv)"
	@echo "  make setup-aws      - Setup AWS environment"
	@echo ""
	@echo "Training commands:"
	@echo "  make train-all      - Train all models"
	@echo "  make train-det      - Train text detection models"
	@echo "  make train-rec      - Train text recognition models"
	@echo "  make train-layout   - Train layout detection models"
	@echo "  make train-table    - Train table recognition model"
	@echo "  make train-lora     - Train LoRA adapter for LLM"
	@echo ""
	@echo "Evaluation commands:"
	@echo "  make eval-all       - Run all evaluations"
	@echo "  make eval-det       - Evaluate detection models"
	@echo "  make eval-llm       - Evaluate LLM performance"
	@echo ""
	@echo "Deployment commands:"
	@echo "  make serve          - Start vLLM server"
	@echo "  make ssh            - SSH to EC2 instance"
	@echo ""
	@echo "Infrastructure commands:"
	@echo "  make infra-plan     - Plan infrastructure changes"
	@echo "  make infra-apply    - Apply infrastructure changes"
	@echo "  make infra-destroy  - Destroy infrastructure"
	@echo ""
	@echo "Development commands:"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linters"
	@echo "  make clean          - Clean temporary files"
	@echo ""
	@echo "Git/Issues automation:"
	@echo "  make commit M=\"msg\" [BRANCH=main]                - Commit and push"
	@echo "  make commit-close M=\"msg\" I=\"1,2\" [BRANCH=main] - Commit, push, and close issues via gh"
	@echo "  make close-last                                    - Close issues referenced by last commit"

# Setup commands
setup-local:
	@echo "Setting up local environment..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip setuptools wheel
	./venv/bin/pip install -r requirements.txt
	@echo "Creating directories..."
	mkdir -p data/{raw,processed,samples} weights/{det,rec,layout,table,lora} configs logs results checkpoints
	@echo "Setup complete! Activate with: source venv/bin/activate"

PYVER ?= 3.12

setup-uv:
	@echo "Setting up local environment with uv (.venv, Python $(PYVER))..."
	uv venv --python $(PYVER)
	. .venv/bin/activate && uv pip install -r requirements.txt --upgrade
	@echo "Creating directories..."
	mkdir -p data/{raw,processed,samples} weights/{det,rec,layout,table,lora} configs logs results checkpoints
	@echo "Setup complete! Activate with: source .venv/bin/activate"

setup-aws: check-env
	@echo "Setting up AWS environment..."
	aws s3 mb s3://$(S3_BUCKET) --region $(AWS_REGION) || true
	@echo "AWS setup complete!"

# Training commands
train-all: train-det train-rec train-layout train-table train-lora

train-det:
	@echo "Training text detection models..."
	bash scripts/run_train.sh det

train-rec:
	@echo "Training text recognition models..."
	bash scripts/run_train.sh rec

train-layout:
	@echo "Training layout detection models..."
	bash scripts/run_train.sh layout

train-table:
	@echo "Training table recognition model..."
	bash scripts/run_train.sh table

train-lora:
	@echo "Training LoRA adapter..."
	python -m src.train_doc_ja_lora --config configs/training_lora.yaml

# Data preparation
prepare-data:
	@echo "Preparing training data..."
	# Create test data for evaluation
	python -m src.eval_docqa --create-test
	python -m src.eval_jsonextract --create-test
	python -m src.eval_summary --create-test
	# Mix instruction data
	python -m src.data_mix \
		--general data/general_instructions.jsonl \
		--doc data/doc_instructions.jsonl \
		--output data/mixed_train.jsonl

# Evaluation commands
eval-all:
	@echo "Running all evaluations..."
	bash scripts/run_eval.sh

eval-det:
	@echo "Evaluating detection models..."
	python -m src.eval_detection --config configs/det_eval.yaml

eval-llm: eval-jsquad eval-docqa eval-json eval-summary

eval-jsquad:
	@echo "Evaluating on JSQuAD..."
	python -m src.eval_jsquad --model weights/lora/adapter --base gpt-oss-20B

eval-docqa:
	@echo "Evaluating on DocQA..."
	python -m src.eval_docqa --model weights/lora/adapter --base gpt-oss-20B

eval-json:
	@echo "Evaluating JSON extraction..."
	python -m src.eval_jsonextract --model weights/lora/adapter --base gpt-oss-20B

eval-summary:
	@echo "Evaluating summarization..."
	python -m src.eval_summary --model weights/lora/adapter --base gpt-oss-20B

# Serving commands
serve:
	@echo "Starting vLLM server..."
	bash scripts/run_vllm.sh

serve-dev:
	@echo "Starting development server..."
	python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8001

# Infrastructure commands
infra-plan: check-env
	@echo "Planning infrastructure..."
	cd infra && terraform plan -var="key_name=$(KEY_NAME)"

infra-apply: check-env
	@echo "Applying infrastructure..."
	cd infra && terraform apply -var="key_name=$(KEY_NAME)" -auto-approve

infra-destroy: check-env
	@echo "Destroying infrastructure..."
	cd infra && terraform destroy -var="key_name=$(KEY_NAME)" -auto-approve

ssh: check-env
	@echo "Connecting to EC2 instance..."
	@INSTANCE_IP=$$(cd infra && terraform output -raw instance_public_ip 2>/dev/null); \
	if [ -z "$$INSTANCE_IP" ]; then \
		echo "No instance found. Run 'make infra-apply' first."; \
		exit 1; \
	fi; \
	ssh -i ~/.ssh/$(KEY_NAME).pem ubuntu@$$INSTANCE_IP

# Development commands
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=src --cov-report=html

test-models:
	@echo "Testing model loading..."
	python -m tests.test_models

test-pipeline:
	@echo "Testing pipeline..."
	python -m tests.test_pipeline

lint:
	@echo "Running linters..."
	black src/ tests/ --check
	isort src/ tests/ --check
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf logs/*.log

# Git + Issues (automation)
BRANCH ?= main

commit:
	@git add -A && git commit -m "$(M)" || echo "No changes to commit"
	@git push origin $(BRANCH)

commit-close:
	@bash scripts/gh_commit_close.sh -m "$(M)" -i "$(I)" -b "$(BRANCH)"

close-last:
	@bash scripts/gh_close_from_commit.sh

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t docja:latest .

docker-run:
	@echo "Running Docker container..."
	docker run --gpus all -p 8000:8000 -v $(PWD)/data:/app/data docja:latest

# Utility commands
check-env:
	@if [ -z "$(AWS_PROFILE)" ]; then \
		echo "Error: AWS_PROFILE not set"; \
		exit 1; \
	fi
	@if [ -z "$(KEY_NAME)" ]; then \
		echo "Error: KEY_NAME not set"; \
		exit 1; \
	fi

download-models:
	@echo "Downloading pretrained models..."
	python scripts/download_models.py

benchmark:
	@echo "Running benchmarks..."
	python scripts/benchmark.py

# Installation check
check-install:
	@echo "Checking installation..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@echo "Installation check complete!"

# Quick start
quickstart: setup-local prepare-data
	@echo "Quickstart complete!"
	@echo "Next steps:"
	@echo "1. Activate environment: source venv/bin/activate"
	@echo "2. Train models: make train-all"
	@echo "3. Run inference: docja samples/document.pdf -o results/"
