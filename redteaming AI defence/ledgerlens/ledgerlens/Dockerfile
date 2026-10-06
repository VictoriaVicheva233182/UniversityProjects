FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LEDGERLENS_HOME=/app
WORKDIR /app
RUN useradd --create-home --uid 1000 app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY configs ./configs
RUN mkdir -p output && chown -R app:app /app
USER app
EXPOSE 8000
# Builds the ledger and the report on first start, then serves the workbench.
CMD ["sh", "-c", "[ -f output/report.html ] || ledgerlens all; ledgerlens serve --host 0.0.0.0 --port 8000"]
