FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY knowledge ./knowledge
RUN pip install --no-cache-dir . && useradd -m app && chown -R app /app
USER app
EXPOSE 8000
# One worker: sessions and rate limits are in memory until Redis is added.
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --workers 1"]
