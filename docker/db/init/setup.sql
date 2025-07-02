CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE IF NOT EXISTS Songs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    album VARCHAR(100) NOT NULL,
    spotify_url VARCHAR(512) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS Artists (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL

);

CREATE TABLE IF NOT EXISTS Song_Artist (
    song_id INTEGER REFERENCES Songs(id),
    artist_id INTEGER REFERENCES Artists(id),
    PRIMARY KEY (song_id, artist_id)
);


-- They will likely embed at different rates so that's why these are separated.
CREATE TABLE IF NOT EXISTS Queue_JukeMIR (
    spotify_url VARCHAR(512) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS Queue_Auditus (
    spotify_url VARCHAR(512) UNIQUE NOT NULL
);


CREATE TABLE IF NOT EXISTS Embeddings_JukeMIR (
    song_id INTEGER REFERENCES Songs(id),
    chunk_id INTEGER NOT NULL,
    embedding vector(4800)
);

CREATE TABLE IF NOT EXISTS Embeddings_JukeMIR_PCA_250 (
    song_id INTEGER REFERENCES Songs(id),
    chunk_id INTEGER NOT NULL,
    embedding vector(250)
);

CREATE TABLE IF NOT EXISTS Embeddings_Auditus (
    song_id INTEGER REFERENCES Songs(id),
    chunk_id INTEGER NOT NULL,
    embedding vector(768)
);

CREATE TABLE IF NOT EXISTS Embeddings_Auditus_PCA_250 (
    song_id INTEGER REFERENCES Songs(id),
    chunk_id INTEGER NOT NULL,
    embedding vector(250)
);

-- For vector similarity search.
-- TODO: try different lists/cluster amounts for better performance (if that becomes an issue).
CREATE INDEX IF NOT EXISTS jukemir_vector_idx ON Embeddings_JukeMIR 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);

CREATE INDEX IF NOT EXISTS jukemir_pca_vector_idx ON Embeddings_JukeMIR_PCA_250 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 500);

CREATE INDEX IF NOT EXISTS auditus_vector_idx ON Embeddings_Auditus 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 800);

CREATE INDEX IF NOT EXISTS auditus_pca_vector_idx ON Embeddings_Auditus_PCA_250 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 500);
