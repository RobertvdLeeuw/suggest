from suggester.objects import Model, TrainEvent, Trajectory, Metric, validated

from suggester.metrics.linear import LinUCBValues, CoefficientChange
from suggester.metrics.general import SongsPicked, Reward

import numpy as np
import pandas as pd

class LinUCB(Model):
    name = "LinUCB"
    id = "linucb_v1"

    allowed_metrics = [SongsPicked, LinUCBValues, CoefficientChange, Reward]

    @property
    def param_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "exploration_matrix": {
                    "type": "array",
                    "items": {
                        "type": "array", 
                        "items": {"type": "number"},
                        "minItems": self.n_dim,
                        "maxItems": self.n_dim
                    },
                    "minItems": self.n_dim,
                    "maxItems": self.n_dim
                },
                "feature_coeffs": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": self.n_dim,
                    "maxItems": self.n_dim
                },
                "n_dim": {
                    "type": "integer", 
                    "minimum": 1,
                }
            },
            "required": ["exploration_matrix", "feature_coeffs", "n_dim"]
        }
    
    @property
    def hyperparam_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "alpha": {
                    "type": "number", 
                    "minimum": 0,
                },
                "update_step": {
                    "type": "integer", 
                    "minimum": 1,
                },
                "feature_start": {
                    "type": "number",
                },
                "n_in": {
                    "type": "integer", 
                    "minimum": 0,
                },
                "n_out": {
                    "type": "integer", 
                    "minimum": 1,
                }
            },
            "required": ["alpha", "update_step", "feature_start", "n_in", "n_out"]
        }

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

    def serialize_params(self) -> dict:
        """Serialize current model parameters"""
        return {
            "exploration_matrix": self.exploration_matrix.tolist(),
            "feature_coeffs": self.feature_coeffs.tolist(),
            "n_dim": self.n_dim
        }
    
    def serialize_hyperparams(self) -> dict:
        """Serialize current hyperparameters"""
        return {
            "alpha": self.alpha,
            "update_step": self.update_step,
            "feature_start": self.feature_start,
            "n_in": self.n_in,
            "n_out": self.n_out
        }

    @classmethod
    def load(cls, hyperparams: dict, params: dict):
        """Load model from serialized parameters"""
        # Create instance with hyperparams
        instance = cls(
            metrics=[],  # Will be set separately
            n_dim=params["n_dim"],
            n_in=hyperparams["n_in"], 
            n_out=hyperparams["n_out"],
            alpha=hyperparams["alpha"],
            update_step=hyperparams["update_step"],
            feature_start=hyperparams["feature_start"]
        )
        
        # Load parameters
        instance.exploration_matrix = np.array(params["exploration_matrix"], dtype="float64")
        instance.feature_coeffs = np.array(params["feature_coeffs"], dtype="float64")
        
        return instance

    def process_inner(self, items: np.ndarray) -> np.ndarray:
        self.calc_metrics(TrainEvent.STEP_START, 
                          exploration_matrix=self.exploration_matrix, 
                          feature_coeffs=self.feature_coeffs, 
                          items=items,
                          alpha=self.alpha)

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
        """Update model with observed rewards"""
        if rewards.dtype != "float64":
            rewards = np.array(rewards, dtype="float64")
        
        unique_songs = np.unique(self.update_buffer["song_id"])
        assert len(rewards) == len(unique_songs), \
            f"Rewards are len {len(rewards)} but update buffer contains {len(unique_songs)} songs: " \
            f"\"{', '.join([str(s) for s in unique_songs])}\""

        songs_to_rewards = dict(zip(unique_songs, rewards))
        
        # Calculate metrics for reward observation
        for song_id, reward in songs_to_rewards.items():
            self.calc_metrics(TrainEvent.REWARD_OBSERVED, 
                            reward=reward, 
                            song_id=song_id)
        
        # Update model parameters
        for chunk in self.update_buffer:
            self.exploration_matrix += np.outer(chunk["embedding"], chunk["embedding"])
            self.feature_coeffs += songs_to_rewards[chunk["song_id"]] * chunk["embedding"]

        # Clear update buffer
        self.update_buffer = np.array([], dtype = [
            ('song_id', 'i4'),
            ('chunk_id', 'i4'), 
            ('embedding', f'f4', (self.n_dim,))
        ])

    async def upload_instance(self, trajectory_id: int):
        """Upload model instance with validation"""
        async with get_session() as s:
            from models import ModelInstanceORM
            model_instance = ModelInstanceORM(
                model_id=self.id,
                trajectory_id=trajectory_id,
                params=validated(self.serialize_params(), self.param_schema, "LinUCB parameters"),
                hyperparams=validated(self.serialize_hyperparams(), self.hyperparam_schema, "LinUCB hyperparameters")
            )
            s.add(model_instance)
            await s.commit()


# Test function
import asyncio
async def test():
    from db import get_embeddings
    from models import EmbeddingAuditus

    emb = await get_embeddings(EmbeddingAuditus)
    l = LinUCB(metrics=[SongsPicked, CoefficientChange], 
               n_dim=768, n_in=0, n_out=1, alpha=0.1, 
               update_step=25, feature_start=-2)

    print("Testing LinUCB model...")
    for i in range(5):
        out = l.process(emb)
        print(f"Step {i+1} - Selected song: {out['song_id']}")
        l.update(np.array([-100]))

    print(f"Coefficient changes: {l.metrics[1].values}")
    
    # Test serialization
    params = l.serialize_params()
    hyperparams = l.serialize_hyperparams()
    print(f"Serialization successful - params keys: {list(params.keys())}")
    print(f"Hyperparams: {hyperparams}")

if __name__ == "__main__":
    asyncio.run(test())
