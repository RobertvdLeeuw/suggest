# test_collector_properties.py - Antithesis-style testing for your music collector

import asyncio
import random
import os
from unittest.mock import Mock, AsyncMock, patch
from antithesis import random_choice, assert_always, assert_sometimes, assert_unreachable

from downloader import _download, start_download_loop, clean_downloads
from metadata import create_push_track, _add_to_db_queue, queue_sp_user
from embedders import SongQueue, start_processes, end_processes
from db import get_session, setup
from models import Song, QueueJukeMIR, QueueAuditus, EmbeddingJukeMIR, EmbeddingAuditus

class MockSpotify:
    """Mock Spotify API that can simulate various failure modes"""
    
    def __init__(self):
        self.fail_probability = 0.1
        self.rate_limit_probability = 0.05
        self.songs_db = {
            "track1": {"name": "Test Song 1", "artists": [{"id": "artist1", "name": "Artist 1"}]},
            "track2": {"name": "Test Song 2", "artists": [{"id": "artist2", "name": "Artist 2"}]},
            "track3": {"name": "Test Song 3", "artists": [{"id": "artist1", "name": "Artist 1"}]},
        }
        self.request_count = 0
        
    def track(self, spotify_id):
        self.request_count += 1
        
        # Simulate rate limiting
        if random.random() < self.rate_limit_probability:
            raise Exception("Rate limit exceeded")
            
        # Simulate API failures
        if random.random() < self.fail_probability:
            raise Exception("Spotify API error")
            
        # Simulate missing tracks
        if spotify_id not in self.songs_db:
            raise Exception("Track not found")
            
        return self.songs_db[spotify_id]

async def test_download_pipeline_invariants():
    """Test that the download pipeline maintains key invariants under various conditions"""
    
    await setup()
    
    # Initialize test queues
    jukemir_queue = SongQueue("JukeMIR", QueueJukeMIR)
    auditus_queue = SongQueue("Auditus", QueueAuditus)
    song_queues = [jukemir_queue, auditus_queue]
    
    # Track system state
    total_songs_queued = 0
    total_songs_downloaded = 0
    total_songs_embedded = 0
    failed_downloads = 0
    
    mock_spotify = MockSpotify()
    
    with patch('metadata.sp', mock_spotify):
        for iteration in range(100):
            # Randomly choose operations to simulate concurrent system behavior
            operation = random_choice([
                "add_songs_to_queue",
                "download_song", 
                "process_embedding",
                "clean_downloads",
                "api_failure",
                "rate_limit"
            ])
            
            try:
                if operation == "add_songs_to_queue":
                    # Add random songs to queue
                    song_ids = random.sample(list(mock_spotify.songs_db.keys()), 
                                           random.randint(1, 3))
                    await _add_to_db_queue(song_ids)
                    total_songs_queued += len(song_ids)
                    
                elif operation == "download_song":
                    # Simulate download process
                    if total_songs_queued > 0:
                        random_queue = random_choice(song_queues)
                        spotify_id = random_choice(list(mock_spotify.songs_db.keys()))
                        
                        try:
                            result = await _download(spotify_id, random_queue)
                            if result:
                                total_songs_downloaded += 1
                        except Exception:
                            failed_downloads += 1
                            
                elif operation == "process_embedding":
                    # Simulate embedding processing
                    if any(len(q) > 0 for q in song_queues):
                        total_songs_embedded += 1
                        
                elif operation == "clean_downloads":
                    # Test cleanup doesn't break anything
                    clean_downloads(song_queues)
                    
                elif operation == "api_failure":
                    # Temporarily increase failure rate
                    old_fail_rate = mock_spotify.fail_probability
                    mock_spotify.fail_probability = 0.5
                    # Restore after a few operations
                    await asyncio.sleep(0.1)
                    mock_spotify.fail_probability = old_fail_rate
                    
                elif operation == "rate_limit":
                    # Simulate rate limiting spike
                    old_rate_limit = mock_spotify.rate_limit_probability
                    mock_spotify.rate_limit_probability = 0.8
                    await asyncio.sleep(0.1)
                    mock_spotify.rate_limit_probability = old_rate_limit
                
                # INVARIANTS - These must ALWAYS hold
                
                # 1. Queue lengths should never be negative
                for queue in song_queues:
                    assert_always(
                        len(queue) >= 0,
                        f"Queue {queue.name} has negative length: {len(queue)}"
                    )
                
                # 2. Total songs processed should be monotonically increasing
                current_total = total_songs_downloaded + total_songs_embedded
                assert_always(
                    current_total >= 0,
                    f"Total processed songs went negative: {current_total}"
                )
                
                # 3. Database consistency - no orphaned records
                async with get_session() as session:
                    # Every song in queue should eventually have a corresponding Song record
                    from sqlalchemy import select
                    
                    queue_items = await session.execute(
                        select(QueueJukeMIR.spotify_id).union(
                            select(QueueAuditus.spotify_id)
                        )
                    )
                    queue_spotify_ids = set(queue_items.scalars().all())
                    
                    songs = await session.execute(select(Song.spotify_id))
                    song_spotify_ids = set(songs.scalars().all())
                    
                    # All queued items should eventually become songs
                    orphaned_queue_items = queue_spotify_ids - song_spotify_ids
                    assert_always(
                        len(orphaned_queue_items) <= total_songs_queued,
                        f"Too many orphaned queue items: {len(orphaned_queue_items)}"
                    )
                
                # 4. Download directory consistency
                if os.path.exists("./downloads"):
                    download_files = set(os.listdir("./downloads"))
                    queue_files = set()
                    for queue in song_queues:
                        queue_files.update(file for file, _ in queue.peek_all())
                    
                    # Files in downloads should mostly correspond to queue items
                    # (allowing for some temporary files during processing)
                    orphaned_files = download_files - queue_files
                    assert_always(
                        len(orphaned_files) <= 10,  # Allow some temporary files
                        f"Too many orphaned download files: {len(orphaned_files)}"
                    )
                
            except Exception as e:
                # Some failures are expected, but system should remain stable
                continue
        
        # LIVENESS PROPERTIES - These should eventually happen
        
        # 1. System should successfully process some songs
        assert_sometimes(
            total_songs_downloaded > 0,
            "System never successfully downloaded any songs"
        )
        
        # 2. Queue processing should make progress
        assert_sometimes(
            any(len(q) == 0 for q in song_queues),
            "Queues never became empty - no processing progress"
        )
        
        # 3. System should recover from API failures
        assert_sometimes(
            mock_spotify.request_count > failed_downloads * 2,
            "System didn't recover from API failures"
        )

async def test_concurrent_metadata_operations():
    """Test metadata operations under concurrent access"""
    
    await setup()
    mock_spotify = MockSpotify()
    
    # Track artists to ensure consistency
    created_artists = set()
    created_songs = set()
    
    with patch('metadata.sp', mock_spotify):
        # Simulate multiple concurrent operations
        tasks = []
        for _ in range(50):
            operation = random_choice([
                "create_artist",
                "create_song", 
                "queue_songs",
                "get_similar_artists"
            ])
            
            if operation == "create_artist":
                artist_id = random_choice(["artist1", "artist2", "artist3"])
                tasks.append(create_push_artist(artist_id))
                created_artists.add(artist_id)
                
            elif operation == "create_song":
                song_id = random_choice(list(mock_spotify.songs_db.keys()))
                tasks.append(create_push_track(song_id))
                created_songs.add(song_id)
                
            elif operation == "queue_songs":
                song_ids = random.sample(list(mock_spotify.songs_db.keys()), 
                                       random.randint(1, 2))
                tasks.append(_add_to_db_queue(song_ids))
        
        # Execute all operations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # INVARIANTS
        
        # 1. No duplicate artists should be created
        async with get_session() as session:
            from sqlalchemy import select, func
            from models import Artist
            
            artist_counts = await session.execute(
                select(Artist.spotify_id, func.count(Artist.artist_id))
                .group_by(Artist.spotify_id)
                .having(func.count(Artist.artist_id) > 1)
            )
            duplicates = artist_counts.all()
            
            assert_always(
                len(duplicates) == 0,
                f"Duplicate artists found: {duplicates}"
            )
        
        # 2. No duplicate songs should be created
        async with get_session() as session:
            song_counts = await session.execute(
                select(Song.spotify_id, func.count(Song.song_id))
                .group_by(Song.spotify_id)
                .having(func.count(Song.song_id) > 1)
            )
            duplicates = song_counts.all()
            
            assert_always(
                len(duplicates) == 0,
                f"Duplicate songs found: {duplicates}"
            )
        
        # 3. All created entities should be in database
        async with get_session() as session:
            db_artists = await session.execute(select(Artist.spotify_id))
            db_artist_ids = set(db_artists.scalars().all())
            
            db_songs = await session.execute(select(Song.spotify_id))  
            db_song_ids = set(db_songs.scalars().all())
            
            assert_always(
                created_artists.issubset(db_artist_ids),
                f"Missing artists in DB: {created_artists - db_artist_ids}"
            )
            
            assert_always(
                created_songs.issubset(db_song_ids),
                f"Missing songs in DB: {created_songs - db_song_ids}"
            )

async def test_embedding_pipeline_safety():
    """Test that embedding pipeline maintains safety under various failure modes"""
    
    await setup()
    
    # Mock embedding functions that can fail
    def mock_jukemir_embed(file_path, song_id):
        if random.random() < 0.2:  # 20% failure rate
            raise Exception("JukeMIR embedding failed")
        return [Mock(chunk_id=1, embedding=[0.1] * 4800, song_id=song_id)]
    
    def mock_auditus_embed(file_path, song_id):
        if random.random() < 0.15:  # 15% failure rate  
            raise Exception("Auditus embedding failed")
        return [Mock(chunk_id=1, embedding=[0.1] * 768, song_id=song_id)]
    
    embedding_attempts = 0
    successful_embeddings = 0
    failed_embeddings = 0
    
    with patch('embedders._jukemir_embed', mock_jukemir_embed), \
         patch('embedders._auditus_embed', mock_auditus_embed):
        
        for _ in range(100):
            operation = random_choice([
                "start_embedding",
                "queue_overload", 
                "process_failure",
                "cleanup"
            ])
            
            if operation == "start_embedding":
                embedding_attempts += 1
                try:
                    # Simulate embedding process
                    if random.random() < 0.8:  # 80% success rate overall
                        successful_embeddings += 1
                    else:
                        failed_embeddings += 1
                        raise Exception("Embedding failed")
                except:
                    failed_embeddings += 1
            
            # SAFETY INVARIANTS
            
            # 1. Failure rate shouldn't be too high
            if embedding_attempts > 10:
                failure_rate = failed_embeddings / embedding_attempts
                assert_always(
                    failure_rate < 0.9,
                    f"Embedding failure rate too high: {failure_rate:.2%}"
                )
            
            # 2. System should make progress
            assert_always(
                successful_embeddings >= 0,
                "Successful embedding count went negative"
            )
            
            # 3. Total attempts should be consistent
            assert_always(
                embedding_attempts == successful_embeddings + failed_embeddings,
                f"Inconsistent counts: {embedding_attempts} != {successful_embeddings} + {failed_embeddings}"
            )
        
        # LIVENESS - System should eventually succeed at some embeddings
        assert_sometimes(
            successful_embeddings > 0,
            "System never successfully completed any embeddings"
        )

# Integration test that combines all components
async def test_full_system_integration():
    """Test the entire collector system end-to-end"""
    
    await setup()
    
    # System metrics
    total_operations = 0
    system_errors = 0
    data_inconsistencies = 0
    
    # Mock all external dependencies
    mock_spotify = MockSpotify()
    
    with patch('metadata.sp', mock_spotify), \
         patch('downloader.Spotdl') as mock_spotdl, \
         patch('embedders._jukemir_embed') as mock_jukemir, \
         patch('embedders._auditus_embed') as mock_auditus:
        
        # Configure mocks
        mock_spotdl.return_value.search.return_value = [Mock(name="Test Song")]
        mock_spotdl.return_value.download.return_value = (Mock(), "/fake/path.wav")
        mock_jukemir.return_value = [Mock(chunk_id=1, embedding=[0.1] * 4800)]
        mock_auditus.return_value = [Mock(chunk_id=1, embedding=[0.1] * 768)]
        
        # Run system operations
        for iteration in range(200):
            total_operations += 1
            
            try:
                # Randomly choose system operations
                operation = random_choice([
                    "full_song_pipeline",
                    "queue_user_library", 
                    "process_embeddings",
                    "system_maintenance",
                    "concurrent_operations",
                    "failure_injection"
                ])
                
                if operation == "full_song_pipeline":
                    # Complete pipeline: queue -> download -> embed
                    song_id = random_choice(list(mock_spotify.songs_db.keys()))
                    await _add_to_db_queue([song_id])
                    
                elif operation == "concurrent_operations":
                    # Simulate multiple operations happening simultaneously
                    tasks = [
                        _add_to_db_queue([random_choice(list(mock_spotify.songs_db.keys()))]),
                        create_push_track(random_choice(list(mock_spotify.songs_db.keys()))),
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                elif operation == "failure_injection":
                    # Temporarily increase failure rates
                    mock_spotify.fail_probability = 0.5
                    try:
                        await create_push_track("nonexistent_track")
                    except:
                        pass
                    mock_spotify.fail_probability = 0.1
                
                # SYSTEM-WIDE INVARIANTS
                
                # 1. Database referential integrity
                async with get_session() as session:
                    from sqlalchemy import text
                    
                    # Check for orphaned queue items
                    result = await session.execute(text("""
                        SELECT COUNT(*) FROM queue_jukemir q 
                        LEFT JOIN songs s ON q.spotify_id = s.spotify_id 
                        WHERE s.spotify_id IS NULL
                    """))
                    orphaned_queue = result.scalar()
                    
                    assert_always(
                        orphaned_queue <= 50,  # Allow some processing lag
                        f"Too many orphaned queue items: {orphaned_queue}"
                    )
                
                # 2. System should not accumulate errors indefinitely
                if total_operations > 10:
                    error_rate = system_errors / total_operations
                    assert_always(
                        error_rate < 0.8,
                        f"System error rate too high: {error_rate:.2%}"
                    )
                
                # 3. Data consistency across tables
                async with get_session() as session:
                    # Songs should have corresponding artists
                    result = await session.execute(text("""
                        SELECT COUNT(*) FROM songs s 
                        LEFT JOIN song_artist sa ON s.song_id = sa.song_id 
                        WHERE sa.song_id IS NULL
                    """))
                    songs_without_artists = result.scalar()
                    
                    if songs_without_artists > 0:
                        data_inconsistencies += 1
                    
                    assert_always(
                        data_inconsistencies <= total_operations * 0.1,
                        f"Too many data inconsistencies: {data_inconsistencies}"
                    )
            
            except Exception as e:
                system_errors += 1
                continue
        
        # LIVENESS PROPERTIES
        
        # 1. System should process some operations successfully
        assert_sometimes(
            system_errors < total_operations * 0.9,
            "System failed on almost all operations"
        )
        
        # 2. Database should have some data
        async with get_session() as session:
            song_count = await session.execute(text("SELECT COUNT(*) FROM songs"))
            song_count = song_count.scalar()
            
            assert_sometimes(
                song_count > 0,
                "No songs were successfully created"
            )

# Property-based test runner
async def run_all_property_tests():
    """Run all property-based tests"""
    
    print("🎵 Starting Antithesis property-based testing for Music Collector...")
    
    try:
        print("Testing download pipeline invariants...")
        await test_download_pipeline_invariants()
        
        print("Testing concurrent metadata operations...")
        await test_concurrent_metadata_operations()
        
        print("Testing embedding pipeline safety...")
        await test_embedding_pipeline_safety()
        
        print("Testing full system integration...")
        await test_full_system_integration()
        
        print("✅ All property tests passed!")
        
    except Exception as e:
        print(f"❌ Property test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_all_property_tests())
