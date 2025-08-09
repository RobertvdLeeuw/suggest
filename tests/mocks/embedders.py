import numpy as np

class jukemirlib_fake:
    def extract(self, audio: np.array, layers: list[int], 
                meanpool: bool, offset=0.0) -> dict[int, np.array]:
        return {l: np.random.rand(4800) for l in layers}
    
    def load_audio(self, file_path: str, offset=0.0, duration=None) -> np.array:
        return np.array([])

SAMPLE_RATE = 32_000
# from auditus.transform import AudioArray
class AudioArray:  # So we don't have to import leading to loading entire model into memory.
    def __init__(self, a: np.array, sr: int):
        self.a = a
        self.sr = sr
    
    def shape(self): return self.a.shape
    def __len__(self): return len(self.a)
    def __getitem__(self, idx): return self.a[idx]

class auditus_fake:
    def AudioEmbedding(self, return_tensors="pt") -> callable:
        def inner(audio: AudioArray) -> np.array:
            return np.random.rand(3, 768)

        return inner

    class AudioLoader:
        def load_audio(self, file_path: str) -> AudioArray:
            return AudioArray(a=np.random.rand(2, 100_000), sr=SAMPLE_RATE)

import os
from spotdl.types.song import Song
import soundfile as sf


class Spotdl_fake:
    def __init__(self, no_cache: bool, spotify_client=None, downloader_settings=None, loop=None):
        pass

    def search(self, spotify_urls: list[str]): 
        return [
            Song(name='Halleluhwah',
                 artists=['CAN'],
                 artist='CAN',
                 genres=['krautrock',
                         'space rock',
                         'progressive rock',
                         'experimental',
                         'psychedelic rock',
                         'art rock',
                         'proto-punk'],
                 disc_number=1,
                 disc_count=1,
                 album_name='Tago Mago (Remastered)',
                 album_artist='CAN',
                 duration=1112,
                 year=1971,
                 date='1971-01-01',
                 track_number=4,
                 tracks_count=7,
                 song_id='0E5fRGNiCT8C0C6BNPb3qh',
                 explicit=False,
                 publisher='Mute',
                 url='https://open.spotify.com/track/0E5fRGNiCT8C0C6BNPb3qh',
                 isrc='DEX170420604',
                 cover_url='https://i.scdn.co/image/ab67616d0000b2734c8df6e776670be529fef096',
                 copyright_text='2004 2011 Spoon Records',
                 download_url=None,
                 lyrics=None,
                 popularity=28,
                 album_id='5iWtrhvwOCWnHN14rqub04',
                 list_name=None,
                 list_url=None,
                 list_position=None,
                 list_length=None,
                 artist_id='4l8xPGtl6DHR2uvunqrl8r',
                 album_type='album',
                 spotify_id=spotify_url.replace("https://open.spotify.com/track/", ""))  # For unique filenames later.
            for spotify_url in spotify_urls
        ]

    def download(self, song: Song) -> tuple[Song, str]:
        mock_path = "./mock_downloads"

        if not os.path.isdir(mock_path):
            os.mkdir(mock_path)

        file_path = f"{mock_path}/{song.artist} - {song.name}.wav"

        duration_s = 30.0 
        num_samples = int(duration_s * SAMPLE_RATE)
        
        # Random audio (stereo)
        audio_data = np.random.uniform(-0.1, 0.1, (num_samples, 2)).astype(np.float32)
        
        sf.write(file_path, audio_data, SAMPLE_RATE)

        return song, file_path

