CREATE EXTENSION IF NOT EXISTS vector;

-- Types (must be created before tables that use them)
CREATE TYPE metadata_enum AS ENUM ('tag', 'genre');

-- Tables
CREATE TABLE IF NOT EXISTS Songs (
    song_id serial PRIMARY KEY,
    song_name varchar(255) NOT NULL,
    album varchar(255) NOT NULL,
    spotify_url varchar(512) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS Artists (
    artist_id serial PRIMARY KEY,
    artist_name varchar(255) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS Song_Artist (
    song_id integer REFERENCES Songs(song_id) ON UPDATE CASCADE ON DELETE CASCADE,
    artist_id integer REFERENCES Artists(artist_id) ON UPDATE CASCADE ON DELETE CASCADE,
    PRIMARY KEY (song_id, artist_id)
);

CREATE TABLE IF NOT EXISTS Metadata (
    metadata_id serial PRIMARY KEY,
    song_id integer REFERENCES Songs(song_id),
    type metadata_enum DEFAULT 'genre',
    value varchar(100)
);

CREATE TABLE IF NOT EXISTS Users (
    user_id serial PRIMARY KEY,
    username varchar(255)
);

CREATE TABLE IF NOT EXISTS Models (
    model_id serial PRIMARY KEY,
    model_name varchar(100)
);

CREATE TABLE IF NOT EXISTS Metrics (
    metric_id serial PRIMARY KEY,
    metric_name varchar(255),
    parent_id integer NULL
);

CREATE TABLE IF NOT EXISTS Model_Performance (
    model_id integer REFERENCES Models(model_id),
    metric_id integer REFERENCES Metrics(metric_id),
    timestamp serial PRIMARY KEY,
    song varchar(255) NULL,
    value numeric
);

-- Queue tables
CREATE TABLE IF NOT EXISTS Queue_JukeMIR (
    spotify_url varchar(512) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS Queue_Auditus (
    spotify_url varchar(512) UNIQUE NOT NULL
);

-- Embedding tables
CREATE TABLE IF NOT EXISTS Embeddings_JukeMIR (
    song_id integer REFERENCES Songs(song_id),
    chunk_id integer NOT NULL,
    embedding vector(4800)
);

CREATE TABLE IF NOT EXISTS Embeddings_JukeMIR_PCA_250 (
    song_id integer REFERENCES Songs(song_id),
    chunk_id integer NOT NULL,
    embedding vector(250)
);

CREATE TABLE IF NOT EXISTS Embeddings_Auditus (
    song_id integer REFERENCES Songs(song_id),
    chunk_id integer NOT NULL,
    embedding vector(768)
);

CREATE TABLE IF NOT EXISTS Embeddings_Auditus_PCA_250 (
    song_id integer REFERENCES Songs(song_id),
    chunk_id integer NOT NULL,
    embedding vector(250)
);

-- Vector similarity search indexes
-- Indexing only allowed for <= 2000 dimensions.
-- CREATE INDEX IF NOT EXISTS jukemir_vector_idx ON Embeddings_JukeMIR 
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 1000);

CREATE INDEX IF NOT EXISTS jukemir_pca_vector_idx ON Embeddings_JukeMIR_PCA_250 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 500);

CREATE INDEX IF NOT EXISTS auditus_vector_idx ON Embeddings_Auditus 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 800);

CREATE INDEX IF NOT EXISTS auditus_pca_vector_idx ON Embeddings_Auditus_PCA_250 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 500);
