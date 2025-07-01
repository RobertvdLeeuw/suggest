# Use the official PostgreSQL image as base
FROM postgres:16

# Install build dependencies for pgvector
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        postgresql-server-dev-16 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone and build pgvector
RUN git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && cd .. \
    && rm -rf pgvector

# Clean up build dependencies
RUN apt-get update \
    && apt-get remove -y \
        build-essential \
        postgresql-server-dev-16 \
        git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
