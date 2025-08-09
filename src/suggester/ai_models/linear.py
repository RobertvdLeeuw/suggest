from suggester.objects import Model, TrainEvent, Trajectory, Metric

from suggester.metrics.linear import LinUCBValues, CoefficientChange
from suggester.metrics.general import SongsPicked

import numpy as np
import pandas as pd

class LinUCB(Model):
    name = "LinUCB"

    allowed_metrics = [SongsPicked, LinUCBValues, CoefficientChange]
    # allowed_rewards = []

    param_schema = {} # TODO

    def __init__(self, metrics: list[Metric],
                 n_dim: int, n_in: int, n_out: int, 
                 alpha: float, update_step: int, feature_start: float):
        assert alpha > 0
        super().__init__(metrics, n_out, n_in)

        self.n_dim = n_dim

        self.feature_start = float(feature_start)

        self.alpha = alpha
        self.update_step = update_step
        self.update_buffer = np.array([], dtype = [
            ('song_id', 'i4'),
            ('chunk_id', 'i4'), 
            ('embedding', f'f4', (self.n_dim,))
        ])

        self.exploration_matrix = np.identity(n_dim, dtype="float64")  # A
        self.feature_coeffs = np.full(n_dim, feature_start, dtype="float64")  # b

    def process_inner(self, items: np.ndarray) -> np.ndarray:
        self.calc_metrics(TrainEvent.STEP_START, 
                          exploration_matrix=self.exploration_matrix, 
                          feature_coeffs=self.feature_coeffs, 
                          items=items)

        explor_inv = np.linalg.inv(self.exploration_matrix)
        coeff = explor_inv @ self.feature_coeffs  # Theta 

        uncertainty_terms = np.sum(items["embedding"] @ explor_inv * items["embedding"], axis=1)
        predicted_rewards = items["embedding"] @ coeff
        ucb_values = predicted_rewards + self.alpha * np.sqrt(uncertainty_terms)

        # More efficient groupby using bincount
        unique_songs, inverse_indices, counts = np.unique(
            items["song_id"], return_inverse=True, return_counts=True
        )
        
        song_ucb_avgs = np.bincount(inverse_indices, weights=ucb_values) / counts

        top_song_indices = np.argpartition(song_ucb_avgs, -self.n_out)[-self.n_out:]
        top_song_ids = unique_songs[top_song_indices]
        
        picked = items[np.isin(items["song_id"], top_song_ids)]
        self.update_buffer = np.concatenate([self.update_buffer, picked])

        self.calc_metrics(TrainEvent.STEP_END, 
                          picked=picked["song_id"])
        return picked

    def update(self, rewards: np.array):
        if rewards.dtype != "float64":
            rewards = np.array(rewards, dtype="float64")
        
        unique_songs = np.unique(self.update_buffer["song_id"])
        assert len(rewards) == len(unique_songs), F"Rewards are len {len(rewards)} but update buffer contains {len(unique_songs)} songs: \"{", ".join([str(s) for s in unique_songs])}\"."

        songs_to_rewards = dict(zip(unique_songs, rewards))
        for chunk in self.update_buffer:
            self.exploration_matrix += np.outer(chunk["embedding"], chunk["embedding"])
            self.feature_coeffs += songs_to_rewards[chunk["song_id"]] * chunk["embedding"]

        self.update_buffer = np.array([], dtype = [
            ('song_id', 'i4'),
            ('chunk_id', 'i4'), 
            ('embedding', f'f4', (self.n_dim,))
        ])
        

import asyncio
async def test():
    from db import setup, get_embeddings
    from models import EmbeddingAuditus

    await setup()
    emb = await get_embeddings(EmbeddingAuditus)
    l = LinUCB(metrics=[SongsPicked, CoefficientChange], 
               n_dim=768, n_in=0, n_out=1, alpha=0.1, 
               update_step=25, feature_start=-2)

    for _ in range(5):
        out = l.process(emb)
        print(out["song_id"])
        l.update(np.array([-100]))

    print(l.metrics[1].values)

if __name__ == "__main__":
    asyncio.run(test())
