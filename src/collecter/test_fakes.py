import numpy as np

class jukemirlib_fake:
    def extract(self, audio: np.array, duration: float, offset=0.0) -> np.array:
        return np.random.rand(4800)
    
    def load_audio(self, file_path: str, offset=0.0, duration=None) -> np.array:
        return np.array([])


from auditus.transform import AudioArray
class auditus_fake:
    def AudioEmbedding(self, return_tensors="pt") -> callable:
        def inner(audio: AudioArray) -> np.array:
            return np.random.rand(3, 768)

        return inner
