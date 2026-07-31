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

# Preserves src/'s layout (collecter/ as a package alongside db.py, models.py,
# logger.py) - collecter/main.py and friends use relative imports (`from . import
# services`), which only resolve if collecter/ stays a package under /app, not
# flattened into it.
COPY src/ .

CMD ["python", "-m", "collecter.main"]
