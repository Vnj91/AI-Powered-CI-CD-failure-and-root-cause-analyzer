FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /home/appuser/app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=5s --timeout=3s --start-period=20s --retries=6 \
    CMD curl -f http://127.0.0.1:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.address=0.0.0.0"]
