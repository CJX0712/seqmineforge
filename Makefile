# SeqMineForge developer tasks. Author: 晨星.
#
# Every target below is a HARD gate. None of them ends in `|| true`: a lint step
# that cannot fail is a lint step that has already stopped working.

PYTHON ?= python
RUFF    ?= $(PYTHON) -m ruff
OUT     ?= benchmark.json

.DEFAULT_GOAL := help
.PHONY: help install dev lint fmt test cov demo verify clean docker secrets all

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

dev:  ## Install package with dev extras (editable)
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## ruff check (hard gate)
	$(RUFF) check .

fmt:  ## Apply ruff formatting
	$(RUFF) format .

fmt-check:  ## Verify formatting (hard gate)
	$(RUFF) format --check .

test:  ## Run the test suite
	$(PYTHON) -m pytest -q -W ignore::UserWarning

cov:  ## Run tests with coverage
	$(PYTHON) -m pytest -q -W ignore::UserWarning --cov=seqmineforge --cov-report=term

demo:  ## Run the full benchmark; exits 1 if any gate fails
	$(PYTHON) examples/run_demo.py --out $(OUT)

verify:  ## Verify the flagship against the oracle on a small database
	$(PYTHON) -m seqmineforge.cli verify --dataset grouped --max-length 3

cli:  ## Print the effective configuration and registry
	$(PYTHON) -m seqmineforge.cli info

# Reproducibility gate: the core metrics must be byte-identical across runs.
determinism:  ## Two runs must agree on every non-timing metric
	@$(PYTHON) examples/run_demo.py --out .det-a.json --quiet
	@$(PYTHON) examples/run_demo.py --out .det-b.json --quiet
	@$(PYTHON) -c "import json,pathlib,sys; a=json.loads(pathlib.Path('.det-a.json').read_text(encoding='utf-8')); b=json.loads(pathlib.Path('.det-b.json').read_text(encoding='utf-8')); ks=('datasets','efficiency','verification','utility','gates'); bad=[k for k in ks if json.dumps(a[k],sort_keys=True)!=json.dumps(b[k],sort_keys=True)]; sys.exit(1) if bad else None; print('determinism OK')"
	@rm -f .det-a.json .det-b.json

secrets:  ## Fail if anything credential-shaped is tracked
	@if git grep -nE "(sk-[A-Za-z0-9]{16,}|api[_-]?key[[:space:]]*[=:]|token[[:space:]]*[=:][[:space:]]*['\"][A-Za-z0-9]{16,}|password[[:space:]]*[=:])" -- '*.py' '*.md' '*.toml' '*.yml'; then \
		echo "potential secret found"; exit 1; \
	else echo "secret scan clean"; fi

all: lint fmt-check test demo secrets  ## Everything CI runs

docker:  ## Build and run the containerised demo
	docker build --target runtime -t seqmineforge:0.1.0 .
	docker run --rm seqmineforge:0.1.0

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	rm -f $(OUT) benchmark-*.json .det-*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete