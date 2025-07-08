from sqlalchemy import (
    Column, Integer, String, Numeric, ForeignKey, 
    Enum as SQLEnum,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector


from enum import Enum


class MetadataType(Enum):
    tag = "tag"
    genre = "genre"

Base = declarative_base()

class Song(Base):
    __tablename__ = 'songs'

    song_id = Column(Integer, primary_key=True, autoincrement=True)
    spotify_id = Column(String(255), unique=True, nullable=False)
    song_name = Column(String(255), nullable=False)
    # album = Column(String(255), nullable=False)

    artists = relationship("Artist", secondary="song_artist", back_populates="songs")
    extra_data = relationship("SongMetadata", back_populates="song", cascade="all, delete-orphan")
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
    extra_data = relationship("ArtistMetadata", back_populates="artist", cascade="all, delete-orphan")

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
    source = Column(String(100))

    artist = relationship("Artist", back_populates="extra_data")

class SongMetadata(Base):
    __tablename__ = 'song_metadata'

    metadata_id = Column(Integer, primary_key=True, autoincrement=True)
    song_id = Column(Integer, ForeignKey('songs.song_id'), nullable=False)
    type = Column(SQLEnum(MetadataType), default=MetadataType.genre)
    value = Column(String(100))

    song = relationship("Song", back_populates="extra_data")

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


class QueueJukeMIR(Base):
    __tablename__ = 'queue_jukemir'

    spotify_id = Column(String(512), unique=True, nullable=False, primary_key=True)

class QueueAuditus(Base):
    __tablename__ = 'queue_auditus'

    spotify_id = Column(String(512), unique=True, nullable=False, primary_key=True)


class EmbeddingJukeMIR(Base):
    __tablename__ = 'embeddings_jukemir'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(Vector(4800))

    song = relationship("Song", back_populates="jukemir_embeddings")

class EmbeddingJukeMIRPCA250(Base):
    __tablename__ = 'embeddings_jukemir_pca_250'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(Vector(250))

    song = relationship("Song", back_populates="jukemir_pca_embeddings")

class EmbeddingAuditus(Base):
    __tablename__ = 'embeddings_auditus'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(Vector(768))

    song = relationship("Song", back_populates="auditus_embeddings")

class EmbeddingAuditusPCA250(Base):
    __tablename__ = 'embeddings_auditus_pca_250'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(Vector(250))

    song = relationship("Song", back_populates="auditus_pca_embeddings")

