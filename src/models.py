from sqlalchemy import (
    Column, Integer, String, Numeric, ForeignKey, DateTime, Boolean,
    Enum as SQLEnum, Float, Time, JSON, Index,
    CheckConstraint, UniqueConstraint, ForeignKeyConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from pgvector.sqlalchemy import Vector

from enum import Enum


class MetadataType(Enum):
    tag = "tag"
    genre = "genre"

class StartEndReason(Enum):
    selected = "selected"
    skipped = "skipped"
    trackdone = "trackdone"
    restarted = "restarted"
    unknown = "unknown"

class HyperparameterType(Enum):
    f32 = "f32"
    bool = "bool"

Base = declarative_base()

class Song(Base):
    __tablename__ = 'songs'

    song_id = Column(Integer, primary_key=True, autoincrement=True)
    spotify_id = Column(String(255), unique=True, nullable=False)
    song_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())

    artists = relationship("Artist", secondary="song_artist", back_populates="songs")
    extra_data = relationship("SongMetadata", back_populates="song", cascade="all, delete-orphan")
    listens = relationship("Listen", back_populates="song", cascade="all, delete-orphan")
    suggested_songs = relationship("Suggested", back_populates="song", cascade="all, delete-orphan")

    # Embeddings relationships
    jukemir_embeddings = relationship("EmbeddingJukeMIR", back_populates="song", cascade="all, delete-orphan")
    jukemir_pca_embeddings = relationship("EmbeddingJukeMIRPCA250", back_populates="song", cascade="all, delete-orphan")
    auditus_embeddings = relationship("EmbeddingAuditus", back_populates="song", cascade="all, delete-orphan")
    auditus_pca_embeddings = relationship("EmbeddingAuditusPCA250", back_populates="song", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_songs_spotify_id', 'spotify_id'),
        Index('idx_songs_name', 'song_name'),
        Index('idx_songs_created_at', 'created_at'),
    )

class Artist(Base):
    __tablename__ = 'artists'

    artist_id = Column(Integer, primary_key=True, autoincrement=True)
    spotify_id = Column(String(255), unique=True, nullable=False)
    artist_name = Column(String(255), nullable=False)
    entirely_queued = Column(Boolean, default=False)
    similar_queued = Column(Boolean, default=False)

    songs = relationship("Song", secondary="song_artist", back_populates="artists")
    extra_data = relationship("ArtistMetadata", back_populates="artist", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_artists_spotify_id', 'spotify_id'),
        Index('idx_artists_name', 'artist_name'),
        Index('idx_artists_queued_status', 'entirely_queued', 'similar_queued'),
    )

class SongArtist(Base):
    __tablename__ = 'song_artist'

    song_id = Column(Integer, ForeignKey('songs.song_id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)
    artist_id = Column(Integer, ForeignKey('artists.artist_id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)

    __table_args__ = (
        Index('idx_song_artist_song_id', 'song_id'),
        Index('idx_song_artist_artist_id', 'artist_id'),
    )

class ArtistMetadata(Base):
    __tablename__ = 'artist_metadata'

    metadata_id = Column(Integer, primary_key=True, autoincrement=True)
    artist_id = Column(Integer, ForeignKey('artists.artist_id'), nullable=False)
    type = Column(SQLEnum(MetadataType), default=MetadataType.genre)
    value = Column(String(100), nullable=False)
    source = Column(String(100), nullable=False)
    parent_id = Column(Integer, ForeignKey('artist_metadata.metadata_id'))
    
    artist = relationship("Artist", back_populates="extra_data")
    
    parent = relationship("ArtistMetadata", remote_side=[metadata_id], back_populates="children")
    children = relationship("ArtistMetadata", back_populates="parent")

    __table_args__ = (
        Index('idx_artist_metadata_artist_id', 'artist_id'),
        Index('idx_artist_metadata_type', 'type'),
        Index('idx_artist_metadata_value', 'value'),
        Index('idx_artist_metadata_source', 'source'),
        Index('idx_artist_metadata_parent_id', 'parent_id'),

        # Prevent duplicate metadata for same artist
        UniqueConstraint('artist_id', 'type', 'value', 'source', name='uq_artist_metadata'),
    )
class SongMetadata(Base):
    __tablename__ = 'song_metadata'

    metadata_id = Column(Integer, primary_key=True, autoincrement=True)
    song_id = Column(Integer, ForeignKey('songs.song_id'), nullable=False)
    type = Column(SQLEnum(MetadataType), default=MetadataType.genre)
    value = Column(String(100), nullable=False)
    source = Column(String(100), nullable=False)
    parent_id = Column(Integer, ForeignKey('song_metadata.metadata_id'))
    
    song = relationship("Song", back_populates="extra_data")
    
    parent = relationship("SongMetadata", remote_side=[metadata_id], back_populates="children")
    children = relationship("SongMetadata", back_populates="parent")

    __table_args__ = (
        Index('idx_song_metadata_song_id', 'song_id'),
        Index('idx_song_metadata_type', 'type'),
        Index('idx_song_metadata_value', 'value'),
        Index('idx_song_metadata_source', 'source'),
        Index('idx_song_metadata_parent_id', 'parent_id'),
        # Prevent duplicate metadata for same song
        UniqueConstraint('song_id', 'type', 'value', 'source', name='uq_song_metadata'),
    )


class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    spotify_id = Column(String(128), unique=True)
    username = Column(String(255))

    listens = relationship("Listen", back_populates="user", cascade="all, delete-orphan")
    suggested_songs = relationship("Suggested", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_users_spotify_id', 'spotify_id'),
        Index('idx_users_username', 'username'),
    )

class Listen(Base):
    __tablename__ = "listens"

    listen_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, 
                     ForeignKey('users.user_id', 
                                onupdate='CASCADE', 
                                ondelete='CASCADE'), 
                     nullable=False)
    song_id = Column(Integer, 
                     ForeignKey('songs.song_id', 
                                onupdate='CASCADE', 
                                ondelete='CASCADE'), 
                     nullable=False)
    
    listened_at = Column(DateTime, default=func.now())
    ms_played = Column(Integer, nullable=False)
    reason_start = Column(SQLEnum(StartEndReason), 
                          default=StartEndReason.unknown, 
                          nullable=False)
    reason_end = Column(SQLEnum(StartEndReason), 
                        default=StartEndReason.unknown, 
                        nullable=False)
    from_history = Column(Boolean, default=False)

    user = relationship("User", back_populates="listens")
    song = relationship("Song", back_populates="listens")
    chunks = relationship("ListenChunk", back_populates="listen", cascade="all, delete-orphan")

    __table_args__ = (
        # Prevent duplicate listens
        UniqueConstraint('user_id', 'song_id', 'listened_at', name='uq_listen'),
        # Performance indexes
        Index('idx_listens_user_id', 'user_id'),
        Index('idx_listens_song_id', 'song_id'),
        Index('idx_listens_listened_at', 'listened_at'),
        Index('idx_listens_user_listened_at', 'user_id', 'listened_at'),
        Index('idx_listens_from_history', 'from_history'),
        Index('idx_listens_reason_start', 'reason_start'),
        Index('idx_listens_reason_end', 'reason_end'),
        # Data validation
        CheckConstraint('ms_played >= 0', name='chk_listens_ms_played_positive'),
    )

class ListenChunk(Base):
    __tablename__ = 'listen_chunks'

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    listen_id = Column(Integer, ForeignKey('listens.listen_id'), nullable=False)
    from_ms = Column(Integer, nullable=False)
    to_ms = Column(Integer, nullable=False)
    
    listen = relationship("Listen", back_populates="chunks")

    __table_args__ = (
        Index('idx_listen_chunks_listen_id', 'listen_id'),
        Index('idx_listen_chunks_time_range', 'from_ms', 'to_ms'),
        # Validation constraints
        CheckConstraint('from_ms >= 0', name='chk_chunks_from_ms_positive'),
        CheckConstraint('to_ms > from_ms', name='chk_chunks_to_greater_than_from'),
        CheckConstraint('to_ms >= 0', name='chk_chunks_to_ms_positive'),
        # Prevent overlapping chunks for same listen
        UniqueConstraint('listen_id', 'from_ms', 'to_ms', name='uq_listen_chunk_range'),
    )

class Suggested(Base):
    __tablename__ = 'suggested'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), primary_key=True)
    score = Column(Float, nullable=False)  # f32 in schema
    predicted = Column(Float, nullable=False)  # f32 in schema
    suggested_by = Column(Integer, nullable=False)

    song = relationship("Song", back_populates="suggested_songs")
    user = relationship("User", back_populates="suggested_songs")

    __table_args__ = (
        Index('idx_suggested_user_id', 'user_id'),
        Index('idx_suggested_song_id', 'song_id'),
        Index('idx_suggested_score', 'score'),
        Index('idx_suggested_predicted', 'predicted'),
        Index('idx_suggested_by', 'suggested_by'),
        Index('idx_suggested_user_score', 'user_id', 'score'),
        # Validation constraints
        # CheckConstraint('score >= 0 AND score <= 1', name='chk_suggested_score_range'),
        # CheckConstraint('predicted >= 0 AND predicted <= 1', name='chk_suggested_predicted_range'),
    )

class Model(Base):
    __tablename__ = 'models'

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100))
    param_schema = Column(JSON)

    hyperparameters = relationship("Hyperparameter", back_populates="model")
    trajectories = relationship("Trajectory", back_populates="model")

    metrics = relationship("Metric", secondary="model_metric", back_populates="models")

    __table_args__ = (
        Index('idx_models_name', 'model_name'),
        UniqueConstraint('model_name', name='uq_model_name'),
    )

class ModelMetric(Base):
    __tablename__ = 'model_metric'

    model_id = Column(Integer, ForeignKey('models.model_id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)
    metric_id = Column(Integer, ForeignKey('metrics.metric_id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)

    __table_args__ = (
        Index('idx_model_metric_model_id', 'model_id'),
        Index('idx_model_metric_metric_id', 'metric_id'),
    )

class Metric(Base):
    __tablename__ = 'metrics'

    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    type = Column(String(255), nullable=False) 

    performances = relationship("ModelPerformance", back_populates="metric")

    parent_id = Column(Integer, ForeignKey('metrics.metric_id'))
    
    models = relationship("Model", secondary="model_metric", back_populates="metrics")

    parent = relationship("Metric", remote_side=[metric_id], back_populates="children")
    children = relationship("Metric", back_populates="parent")

    __table_args__ = (
        Index('idx_metrics_name', 'name'),
        Index('idx_metrics_type', 'type'),
        Index('idx_metrics_parent_id', 'parent_id'),
        UniqueConstraint('name', 'type', name='uq_metric_name_type'),
    )

class ModelPerformance(Base):
    __tablename__ = 'performances'

    metric_id = Column(Integer, ForeignKey('metrics.metric_id'), primary_key=True)
    trajectory_id = Column(Integer, ForeignKey('trajectory.trajectory_id'), primary_key=True)
    timestep = Column(Integer, primary_key=True)
    song = Column(String(255))
    value = Column(Float, nullable=False)

    metric = relationship("Metric", back_populates="performances")
    trajectory = relationship("Trajectory", back_populates="performances")

    __table_args__ = (
        Index('idx_model_performance_metric_id', 'metric_id'),
        Index('idx_model_performance_trajectory_id', 'trajectory_id'),
        Index('idx_model_performance_timestep', 'timestep'),
        Index('idx_model_performance_song', 'song'),
        Index('idx_model_performance_value', 'value'),
        Index('idx_model_performance_trajectory_timestep', 'trajectory_id', 'timestep'),
        # Validation constraints
        CheckConstraint('timestep >= 0', name='chk_performance_timestep_positive'),
    )

class Funnel(Base):
    __tablename__ = 'funnel'

    funnel_id = Column(Integer, primary_key=True, autoincrement=True)
    funnel_name = Column(String(255), nullable=False)

    models = relationship("FunnelModel", back_populates="funnel")

    __table_args__ = (
        Index('idx_funnel_name', 'funnel_name'),
        UniqueConstraint('funnel_name', name='uq_funnel_name'),
    )

class FunnelModel(Base):
    __tablename__ = 'funnel_model'

    funnel_id = Column(Integer, ForeignKey('funnel.funnel_id'), primary_key=True)
    model_id = Column(Integer, ForeignKey('models.model_id'), primary_key=True)
    position = Column(Integer, nullable=False)

    funnel = relationship("Funnel", back_populates="models")
    model = relationship("Model")

    __table_args__ = (
        Index('idx_funnel_model_funnel_id', 'funnel_id'),
        Index('idx_funnel_model_model_id', 'model_id'),
        Index('idx_funnel_model_position', 'position'),
        # Ensure unique positions within each funnel
        UniqueConstraint('funnel_id', 'position', name='uq_funnel_position'),
        # Validation constraints
        CheckConstraint('position >= 0', name='chk_funnel_model_position_positive'),
    )

class Trajectory(Base):
    __tablename__ = 'trajectory'

    trajectory_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('models.model_id'))
    funnel_id = Column(Integer, ForeignKey('funnel.funnel_id'))

    started = Column(DateTime, nullable=False)
    ended = Column(DateTime)
    timesteps = Column(Integer, nullable=False)
    on_history = Column(Boolean, nullable=False)

    model = relationship("Model", back_populates="trajectories")
    performances = relationship("ModelPerformance", back_populates="trajectory")
    param_instances = relationship("ParamInstance", back_populates="trajectory")

    __table_args__ = (
        Index('idx_trajectory_model_id', 'model_id'),
        Index('idx_trajectory_funnel_id', 'funnel_id'),
        Index('idx_trajectory_started', 'started'),
        Index('idx_trajectory_ended', 'ended'),
        Index('idx_trajectory_on_history', 'on_history'),
        Index('idx_trajectory_model_started', 'model_id', 'started'),
        # Validation constraints
        CheckConstraint('timesteps > 0', name='chk_trajectory_timesteps_positive'),
        CheckConstraint('ended IS NULL OR ended >= started', name='chk_trajectory_end_after_start'),
    )

class ParamInstance(Base):
    __tablename__ = 'param_instances'

    model_id = Column(Integer, ForeignKey('models.model_id'), primary_key=True)
    trajectory_id = Column(Integer, ForeignKey('trajectory.trajectory_id'), primary_key=True)
    params = Column(JSON, nullable=False)  # blob in schema, using JSON for SQLAlchemy

    
    model = relationship("Model")
    trajectory = relationship("Trajectory", back_populates="param_instances")

    __table_args__ = (
        Index('idx_param_instances_model_id', 'model_id'),
        Index('idx_param_instances_trajectory_id', 'trajectory_id'),
    )

class Hyperparameter(Base):
    __tablename__ = 'hyperparameters'

    model_id = Column(Integer, ForeignKey('models.model_id'), primary_key=True)
    hp_id = Column(Integer, primary_key=True)
    type = Column(SQLEnum(HyperparameterType), nullable=False)  # enum (f32, bool)
    min = Column(Float)  # f32 in schema
    max = Column(Float)  # f32 in schema

    model = relationship("Model", back_populates="hyperparameters")
    hp_instances = relationship("HPInstance", back_populates="hyperparameter")

    __table_args__ = (
        Index('idx_hyperparameters_model_id', 'model_id'),
        Index('idx_hyperparameters_type', 'type'),
        # Validation constraints
        CheckConstraint('min IS NULL OR max IS NULL OR min <= max', name='chk_hyperparameter_min_max'),
    )

class HPInstance(Base):
    __tablename__ = 'hp_instances'

    model_id = Column(Integer, ForeignKey('models.model_id'), primary_key=True)
    trajectory_id = Column(Integer, ForeignKey('trajectory.trajectory_id'), primary_key=True)
    hp_id = Column(Integer, primary_key=True)
    type = Column(SQLEnum(HyperparameterType), nullable=False)  # enum (f32, bool)
    value = Column(Float, nullable=False)  # f32 in schema

    trajectory = relationship("Trajectory")
    hyperparameter = relationship("Hyperparameter", back_populates="hp_instances")

    __table_args__ = (
        Index('idx_hp_instances_model_id', 'model_id'),
        Index('idx_hp_instances_trajectory_id', 'trajectory_id'),
        Index('idx_hp_instances_type', 'type'),
        Index('idx_hp_instances_value', 'value'),
        ForeignKeyConstraint(['model_id'], ['models.model_id']),
        ForeignKeyConstraint(['model_id', 'hp_id'], ['hyperparameters.model_id', 'hyperparameters.hp_id']),
    )


class QueueJukeMIR(Base):
    __tablename__ = 'queue_jukemir'

    spotify_id = Column(String(512), unique=True, nullable=False, primary_key=True)
    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_queue_jukemir_created_at', 'created_at'),
    )

class QueueAuditus(Base):
    __tablename__ = 'queue_auditus'

    spotify_id = Column(String(512), unique=True, nullable=False, primary_key=True)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_queue_auditus_created_at', 'created_at'),
    )
    
class EmbeddingJukeMIR(Base):
    __tablename__ = 'embeddings_jukemir'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(Vector(4800), nullable=False)

    song = relationship("Song", back_populates="jukemir_embeddings")

    __table_args__ = (
        Index('idx_embeddings_jukemir_song_id', 'song_id'),
        Index('idx_embeddings_jukemir_chunk_id', 'chunk_id'),
        CheckConstraint('chunk_id >= 0', name='chk_jukemir_chunk_id_positive'),
    )

class EmbeddingJukeMIRPCA250(Base):
    __tablename__ = 'embeddings_jukemir_pca_250'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(Vector(250), nullable=False)

    song = relationship("Song", back_populates="jukemir_pca_embeddings")

    __table_args__ = (
        Index('idx_embeddings_jukemir_pca_song_id', 'song_id'),
        Index('idx_embeddings_jukemir_pca_chunk_id', 'chunk_id'),
        CheckConstraint('chunk_id >= 0', name='chk_jukemir_pca_chunk_id_positive'),
    )

class EmbeddingAuditus(Base):
    __tablename__ = 'embeddings_auditus'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(Vector(768), nullable=False)

    song = relationship("Song", back_populates="auditus_embeddings")

    __table_args__ = (
        Index('idx_embeddings_auditus_song_id', 'song_id'),
        Index('idx_embeddings_auditus_chunk_id', 'chunk_id'),
        CheckConstraint('chunk_id >= 0', name='chk_auditus_chunk_id_positive'),
    )

class EmbeddingAuditusPCA250(Base):
    __tablename__ = 'embeddings_auditus_pca_250'

    song_id = Column(Integer, ForeignKey('songs.song_id'), primary_key=True)
    chunk_id = Column(Integer, nullable=False, primary_key=True)
    embedding = Column(Vector(250), nullable=False)

    song = relationship("Song", back_populates="auditus_pca_embeddings")
    
    __table_args__ = (
        Index('idx_embeddings_auditus_pca_song_id', 'song_id'),
        Index('idx_embeddings_auditus_pca_chunk_id', 'chunk_id'),
        CheckConstraint('chunk_id >= 0', name='chk_auditus_pca_chunk_id_positive'),
    )
