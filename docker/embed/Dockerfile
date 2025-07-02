FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
# COPY poetry.lock* ./ 

# Install dependencies using uv
RUN uv pip install --system -e .

COPY ../../src/embedders/main.py .

CMD ["python", "main.py"]
