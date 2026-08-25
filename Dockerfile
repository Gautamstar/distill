# CPU-only image for the Donut backend.
#
# The zero-shot VLM is 3B parameters and needs a GPU, which on AWS means a
# ~$380/month instance. The fine-tuned Donut is ~200M and runs on CPU, so this
# image fits App Runner, Fargate, or a Lambda container - the difference
# between a demo that costs money to leave running and one that does not.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DISTILL_BACKEND=donut \
    DISTILL_FORCE_CPU=1 \
    DISTILL_DONUT_DIR=/app/model \
    HF_HUB_OFFLINE=1

WORKDIR /app

# The CPU wheel is ~200MB against ~2.5GB for the CUDA build. Installed from the
# CPU index first so the dependency resolver cannot pull the GPU variant in.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
 && pip install --no-cache-dir \
        transformers sentencepiece protobuf pillow pydantic \
        fastapi "uvicorn[standard]" python-multipart

# Weights before source: they change far less often, so edits to the app do not
# invalidate the expensive layer.
COPY model/ /app/model/

COPY distill/ /app/distill/
COPY review/ /app/review/

# Model files are read-only at runtime; uploads and traces are not.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/outputs \
 && chown -R app:app /app/outputs
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "review.app:app", "--host", "0.0.0.0", "--port", "8080"]
