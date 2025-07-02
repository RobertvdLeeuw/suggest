from auditus.transform import AudioArray, AudioLoader, AudioEmbedding, Pooling

import numpy as np


SEGMENT_LENGTH = 24
SEGMENT_OVERLAP = 1
SAMPLE_RATE = 44_100  # Jukebox SR


def embed(file_path: str) -> np.array:
    audio = AudioLoader(sr=SAMPLE_RATE)(file_path)
    
    embeddings = []
    for offset in range(0, len(audio)/SAMPLE_RATE, SEGMENT_LENGTH - SEGMENT_OVERLAP):  # Seconds
        offset_sr = offset * SAMPLE_RATE
        audio_chunk = AudioArray(a=audio.a[offset_sr:offset_sr+SEGMENT_LENGTH *SAMPLE_RATE],
                                 sr=SAMPLE_RATE)

        emb = AudioEmbedding(return_tensors="pt")(audio_chunk)
        emb = Pooling(pooling="mean")(emb)  # JukeMIR meanpools as well, try max at some point?

        embeddings.append(emb.to_numpy())
        
    return np.array(embeddings)
