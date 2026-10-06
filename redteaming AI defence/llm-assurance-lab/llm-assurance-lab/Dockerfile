# API and test bench. The model runs in the separate `ollama` service (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASSURANCE_HOME=/app

WORKDIR /app
RUN useradd --create-home --uid 1000 app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY configs ./configs
COPY data ./data
RUN mkdir -p models reports && assurance-lab train && chown -R app:app /app

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["assurance-lab", "serve", "--host", "0.0.0.0", "--port", "8000"]
