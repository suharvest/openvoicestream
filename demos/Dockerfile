# SLV Demo Gallery — unified demo image (one image, six apps).
#
# Default CMD starts the gallery; every other demo is selected by overriding
# `command:` in docker-compose.demos.yml, e.g.:
#   uv run --frozen python asr-caption/backend/main.py
#
# Build + push (from repo root, arm64 for Jetson/RPi targets):
#   docker buildx build --platform linux/arm64 -f demos/Dockerfile \
#     -t sensecraft-missionpack.seeed.cn/solution/slv-demos:<tag> --push demos/
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV UV_CACHE_DIR=/tmp/uv-cache

# Dependency layer — invalidated only when the lockfile changes.
# (pyproject has no [build-system], so uv installs dependencies only.)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application source — all six demos share the single dependency set above.
COPY common/ common/
COPY gallery/ gallery/
COPY asr-caption/ asr-caption/
COPY tts-playground/ tts-playground/
COPY v2v-chat/ v2v-chat/
COPY diarization/ diarization/
COPY voice-clone/ voice-clone/
COPY registry.json ./

CMD ["uv", "run", "--frozen", "python", "-m", "uvicorn", "gallery.backend.main:app", "--host", "0.0.0.0", "--port", "8700"]
