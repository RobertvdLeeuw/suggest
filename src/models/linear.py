from objects import Model, TrainEvent, Trajectory, Metric, METADATA_COLS

import numpy as np
import pandas as pd


class LinUCB(Model):
    name = "LinUCB"

    def __init__(self, n_dim: int, alpha: float, feature_start: float, description="") -> None:
        assert alpha > 0
        super().__init__()

        if description:
            self.name += f" ({description})"

        self.n_dim = n_dim

        if isinstance(feature_start, int):
            feature_start = float(feature_start)
        self.feature_start = feature_start

        self.alpha = alpha

        self.exploration_matrix: np.ndarray  # A
        self.feature_coeffs: np.ndarray  # b

    def _calc_song_values(self, embeddings: pd.DataFrame, songs_left: list[str]) -> dict[str, float]:
        explor_inv = np.linalg.inv(self.exploration_matrix)
        coeff = explor_inv @ self.feature_coeffs  # Theta

        # embeddings = embeddings[embeddings["name"].isin(songs_left)]  # Slower because pandas.

        all_chunks = embeddings.drop(columns=METADATA_COLS).to_numpy()
        song_indices = embeddings.groupby('name').indices

        uncertainty_terms = np.sum(all_chunks @ explor_inv * all_chunks, axis=1)
        predicted_rewards = all_chunks @ coeff
        ucb_values = predicted_rewards + self.alpha * np.sqrt(uncertainty_terms)

        return {name: np.mean(ucb_values[indices]) for name, indices in song_indices.items()
                if name in songs_left}

    def train(self, embeddings: pd.DataFrame, metrics: list[Metric], T: int) -> Trajectory:
        assert T > 0

        songs_left = list(embeddings["name"].unique())
        n_good_songs = len(embeddings[embeddings.score.isin(["liked", "loved"])]["name"].unique())

        trajectory = Trajectory(metrics={m.name: m(good_songs=n_good_songs) for m in metrics},
                                model_name=self.name, T=T)

        self.exploration_matrix = np.identity(self.n_dim)
        self.feature_coeffs = np.full(self.n_dim, self.feature_start)

        for t in range(T):
            [m.calc(TrainEvent.STEP_START, self.exploration_matrix, self.feature_coeffs, 
                    self.alpha, embeddings) for m in trajectory.metrics.values()]

            song_values = self._calc_song_values(embeddings, songs_left)
            picked = max(song_values, key=song_values.get)
            songs_left.remove(picked)
            
            song_data = embeddings[embeddings["name"] == picked]
            reward = int(song_data.iloc[0]["score"] in["liked", "loved"])

            [m.calc(TrainEvent.REWARD_OBSERVED, reward) for m in trajectory.metrics.values()]
            
            for chunk in song_data.drop(columns=METADATA_COLS).to_numpy():
                self.exploration_matrix += np.outer(chunk, chunk)
                self.feature_coeffs += reward * chunk
           
            [m.calc(TrainEvent.STEP_END, picked) for m in trajectory.metrics.values()]

            if len(songs_left) == 0:
                trajectory.T = t
                break

        return trajectory
