import librosa as lr
import jukemirlib

import os

from math import floor
import numpy as np
import pandas as pd

SEGMENT_OVERLAP = 1 
SEGMENT_LENGTH = 24 

LAYER = 36


def embed(file_path: str) -> np.array:
    length = floor(lr.get_duration(filename=file_path))
    
    embeddings = []

    for offset in range(0, length, SEGMENT_LENGTH - SEGMENT_OVERLAP):
        audio = jukemirlib.load_audio(file_path,  # TODO: This throws errors sometimes, figure out.
                                      offset=offset,
                                      duration=min(SEGMENT_LENGTH, length - offset))
        emb = jukemirlib.extract(audio, layers=[LAYER], meanpool=True)[LAYER]

        embeddings.append(emb)
    return np.array(embeddings)

