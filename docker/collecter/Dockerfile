FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system -e .[collecter,db]

RUN pip install spotdl

COPY src/collecter/* .
COPY src/db.py .
COPY src/models.py .
COPY src/logger.py .

CMD ["python", "main.py"]

