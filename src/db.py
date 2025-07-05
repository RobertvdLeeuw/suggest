from sqlalchemy import (
    create_engine, Column, Integer, String, Numeric, ForeignKey, 
    Text, Index, Enum as SQLEnum, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import VECTOR
from enum import Enum
import os

# Define the metadata enum
class MetadataType(Enum):
    tag = "tag"
    genre = "genre"

Base = declarative_base()

class Song(Base):
    __tablename__ = 'songs'

    song_id = Column(Integer, primary_key=True, autoincrement=True)
    spotify_id = Column(String(255), unique=True, nullable=False)
    song_name = Column(String(255), nullable=False)
    album = Column(String(255), nullable=False)
    spotify_url = Column(String(512), unique=True, nullable=False)

    artists = relationship("Artist", secondary="song_artist", back_populates="songs")
    metadata = relationship("SongMetadata", back_populates="song", cascade="all, delete-orphan")
    jukemir_embeddings = relationship("EmbeddingJukeMIR", back_populates="song", cascade="all, delete-orphan")
    jukemir_pca_embeddings = relationship("EmbeddingJukeMIRPCA250", back_populates="song", cascade="all, delete-orphan")
    auditus_embeddings = relationship("EmbeddingAuditus", back_populates="song", cascade="all, delete-orphan")
    auditus_pca_embeddings = relationship("EmbeddingAuditusPCA250", back_populates="song", cascade="all, delete-orphan")

class Artist(Base):
    __tablename__ = 'artists'

    artist_id = Column(Integer, primary_key=True, autoincrement=True)
    spotify_id = Column(String(255), unique=True, nullable=False)
    artist_name = Column(String(255), unique=True, nullable=False)

    songs = relationship("Song", secondary="song_artist", back_populates="artists")
    metadata = relationship("ArtistMetadata", back_populates="artist", cascade="all, delete-orphan")

class SongArtist(Base):
    __tablename__ = 'song_artist'

    song_id = Column(Integer, ForeignKey('songs.song_id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)
    artist_id = Column(Integer, ForeignKey('artists.artist_id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)

class ArtistMetadata(Base):
    __tablename__ = 'artist_metadata'

    metadata_id = Column(Integer, primary_key=True, autoincrement=True)
    artist_id = Column(Integer, ForeignKey('artists.artist_id'), nullable=False)
    type = Column(SQLEnum(MetadataType), default=MetadataType.genre)
    value = Column(String(100))

    artist = relationship("Artist", back_populates="metadata")

class SongMetadata(Base):
    __tablename__ = 'song_metadata'

    metadata_id = Column(Integer, primary_key=True, autoincrement=True)
    song_id = Column(Integer, ForeignKey('songs.song_id'), nullable=False)
    type = Column(SQLEnum(MetadataType), default=MetadataType.genre)
    value = Column(String(100))

    song = relationship("Song", back_populates="metadata")

class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255))

class Model(Base):
    __tablename__ = 'models'

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100))

    performances = relationship("ModelPerformance", back_populates="model")

class Metric(Base):
    __tablename__ = 'metrics'

    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(255))
    parent_id = Column(Integer, nullable=True)

    performances = relationship("ModelPerformance", back_populates="metric")

class ModelPerformance(Base):
    __tablename__ = 'model_performance'

    timestamp = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('models.model_id'), nullable=False)
    metric_id = Column(Integer, ForeignKey('metrics.metric_id'), nullable=False)
    song = Column(String(255), nullable=True)
    value = Column(Numeric)

    model = relationship("Model", back_populates="performances")
    metric = relationship("Metric", back_populates="performances")

# Queue tables
class QueueJukeMIR(Base):
    __tablename__ = 'queue_jukemir'

    spotify_id = Column(String(512), unique=True, nullable=False, primary_key=True)

class QueueAuditus(Base):
    __tablename__ = 'queue_auditus'

    spotify_id = Column(String(512), unique=True, nullable=False, primary_key=True)

# Embedding tables
class EmbeddingJukeMIR(Base):
    __tablename__ = 'embeddings_jukemir'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(VECTOR(4800))

    song = relationship("Song", back_populates="jukemir_embeddings")

class EmbeddingJukeMIRPCA250(Base):
    __tablename__ = 'embeddings_jukemir_pca_250'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(VECTOR(250))

    song = relationship("Song", back_populates="jukemir_pca_embeddings")

class EmbeddingAuditus(Base):
    __tablename__ = 'embeddings_auditus'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(VECTOR(768))

    song = relationship("Song", back_populates="auditus_embeddings")

class EmbeddingAuditusPCA250(Base):
    __tablename__ = 'embeddings_auditus_pca_250'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(VECTOR(250))

    song = relationship("Song", back_populates="auditus_pca_embeddings")

# Vector similarity search indexes
# Note: These indexes are created via raw SQL as they're specific to pgvector

# Database setup and session management
class DatabaseManager:
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Default connection string - modify as needed
            database_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/music_db')

        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)

    def create_extensions_and_indexes(self):
        """Create PostgreSQL extensions and vector indexes"""
        with self.engine.connect() as conn:
            # Create vector extension
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create custom enum type
            conn.execute("CREATE TYPE metadata_enum AS ENUM ('tag', 'genre');")

            # Create vector indexes (commented out the high-dimensional one as per original SQL)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS jukemir_pca_vector_idx ON embeddings_jukemir_pca_250 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 500);
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS auditus_vector_idx ON embeddings_auditus 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 800);
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS auditus_pca_vector_idx ON embeddings_auditus_pca_250 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 500);
            """)

            conn.commit()

    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()

# Usage example
def main():
    # Initialize database manager
    db_manager = DatabaseManager()

    # Create extensions and tables
    db_manager.create_extensions_and_indexes()
    db_manager.create_tables()

    # Example usage
    session = db_manager.get_session()

    try:
        # Create a new artist
        artist = Artist(
            spotify_id="spotify_artist_123",
            artist_name="Example Artist"
        )
        session.add(artist)
        session.commit()

        # Create a new song
        song = Song(
            spotify_id="spotify_song_456",
            song_name="Example Song",
            album="Example Album",
            spotify_url="https://open.spotify.com/track/example"
        )
        session.add(song)
        session.commit()

        # Link artist and song
        song_artist = SongArtist(song_id=song.song_id, artist_id=artist.artist_id)
        session.add(song_artist)
        session.commit()

        print("Successfully created artist and song!")

    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    main()
