FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

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

# These paths are runtime state. Creating them in the image gives fresh named
# volumes the ownership needed by the non-root application user.
RUN mkdir -p data models output && \
    chown -R appuser:appuser data models output

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=5s --timeout=3s --start-period=20s --retries=6 \
    CMD curl -f "http://127.0.0.1:${PORT}/_stcore/health" || exit 1

ENTRYPOINT ["sh", "-c", "exec streamlit run app.py --server.port=\"${PORT}\" --server.headless=true --server.address=0.0.0.0"]
