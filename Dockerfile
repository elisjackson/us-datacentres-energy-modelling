FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed by geopandas
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY . .

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 "src.main:server"