# Official PyTorch image: torch 2.6 + CUDA 12.4 + Python 3.11 are already inside (no multi-GB pip download of torch)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=15

WORKDIR /app

# project requirements (torch already present; faiss-cpu, mlflow, bitsandbytes, datasets, pytest are not needed by the app)
# one RUN per group: if the flaky network corrupts a download, Docker keeps the groups that already succeeded
RUN python -m pip install "numpy" "scikit-learn>=1.4" "pandas>=2.0" "pyyaml>=6.0" tqdm
RUN python -m pip install "transformers>=4.44" "accelerate>=0.33"
RUN python -m pip install "sentence-transformers>=3.0" "rank_bm25>=0.2.2"
RUN python -m pip install "openai>=1.40" "django>=5.0,<6" "gunicorn>=22.0"
# the conda Brotli (1.1.0) in the base image is too old for the openai client's httpx2 decoder
RUN python -m pip install -U "brotli>=1.2"

# code only; models/ and data/corpus/ are mounted at runtime (see docker-compose.yml)
COPY arroute ./arroute
COPY webapp ./webapp

ENV ARROUTE_ROOT=/app ARROUTE_MODELS=/app/models ARROUTE_CORPUS=/app/data/corpus \
    ARROUTE_LLM_CACHE=/app/data/llm_cache HF_HOME=/root/.cache/huggingface
WORKDIR /app/webapp
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["gunicorn", "config.wsgi", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "300"]
