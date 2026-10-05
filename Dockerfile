# SeqMineForge — verified deterministic sequential pattern mining.
# Author: 晨星
#
# Two stages so the test toolchain never bloats the runtime image.
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is cached unless the pins change.
COPY requirements.txt requirements.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY seqmineforge ./seqmineforge
COPY examples ./examples

RUN python -m pip install --no-cache-dir --no-deps -e .

# Fail the build if the package cannot import or the CLI cannot run -- a broken
# image should never leave the builder.
RUN python -c "import seqmineforge; print('import ok', seqmineforge.__version__)" \
 && python -m seqmineforge.cli info > /dev/null \
 && echo "build-time verification passed"


# ---------------------------------------------------------------- test stage
FROM base AS test
COPY requirements.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements.lock.txt
COPY tests ./tests
COPY .github ./.github
RUN python -m ruff check . \
 && python -m ruff format --check . \
 && python -m pytest -q -W ignore::UserWarning --cov=seqmineforge --cov-report=term


# --------------------------------------------------------------- runtime
FROM base AS runtime
# Non-root: the benchmark needs no writes outside /app.
RUN useradd --create-home --uid 1000 smf && chown -R smf:smf /app
USER smf

# Exits non-zero when any acceptance gate fails.
ENTRYPOINT ["python", "examples/run_demo.py"]
CMD ["--out", "benchmark.json"]