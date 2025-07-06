FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    # gcc \
    # libc-dev \
    # cmake \
    # libxslt-dev \
    # libxml2-dev \
    # libffi-dev \
    # libssl-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system -e .

RUN pip install spotdl

COPY src/collection/* .
# TODO: Something .env loading
COPY src/logger.py .
CMD ["python", "main.py"]
