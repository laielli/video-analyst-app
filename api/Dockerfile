# syntax=docker/dockerfile:1
#
# Runtime image for the Glass-Box Video Analyst API (FastAPI + uvicorn).
#
# Replay-only server: it reads committed cache JSON under api/ (examples/, interpreter/) and
# calls Azure OpenAI for live free-text codegen. It does NOT need ffmpeg, the source .mov
# files, the frame stills, or Azure Vision — those belong to the offline precompute pipeline,
# and the stills are served by the static web app. So the image stays small.
#
# Build context is the api/ directory (keeps the repo's 180MB .mov files out of the context).
# NOTE: az acr build resolves --file relative to the CWD, so build from api/ with context '.':
#   (cd api && az acr build --registry <acr> --image glass-box-api:<tag> --file Dockerfile .)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app/api

# Dependencies first, for layer caching. requirements.txt carries the (small) test deps too;
# installing them as-is keeps the image identical to CI rather than maintaining a runtime split.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code. .dockerignore keeps .venv / __pycache__ / .run_cache / tests out.
COPY . ./

# Non-root runtime user; pre-create the ephemeral run-cache dir it writes to.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/api/.run_cache \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Bind to $PORT; defaults to 8000 to match the Bicep ingress targetPort. Container Apps routes
# to targetPort and does not inject a PORT var, so the 8000 fallback is what actually binds.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
