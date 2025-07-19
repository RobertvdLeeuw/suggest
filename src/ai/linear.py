from objects import Model, TrainEvent, Trajectory, Metric, METADATA_COLS

import numpy as np
import pandas as pd


class LinUCB(Layer):
    name = "LinUCB"

    def __init__(self, metrics: list[Metric]
                 n_dim: int, n_in: int, n_out: int, 
                 alpha: float, update_step: int, feature_start: float) -> None:
        assert alpha > 0
        super().__init__()

        if description:
            self.name += f" ({description})"

        self.n_dim = n_dim

        if isinstance(feature_start, int):
            feature_start = float(feature_start)
        self.feature_start = feature_start

        self.alpha = alpha
        self.update_step = update_step
        self.t = 0

        self.songs_left = []
        self.exploration_matrix = np.identity(n_dim)  # A
        self.feature_coeffs = np.full(n_dim, feature_start)  # b

    def _calc_song_values(self, embeddings: pd.DataFrame, over_all=False) -> dict[str, float]:
        explor_inv = np.linalg.inv(self.exploration_matrix)
        coeff = explor_inv @ self.feature_coeffs  # Theta

        # embeddings = embeddings[embeddings["name"].isin(songs_left)]  # Slower because pandas.

        all_chunks = embeddings.drop(columns=METADATA_COLS).to_numpy()
        song_indices = embeddings.groupby('name').indices

        uncertainty_terms = np.sum(all_chunks @ explor_inv * all_chunks, axis=1)
        predicted_rewards = all_chunks @ coeff
        ucb_values = predicted_rewards + self.alpha * np.sqrt(uncertainty_terms)

        if over_all:
            return {name: np.mean(ucb_values[indices]) for name, indices in song_indices.items()}

        return {name: np.mean(ucb_values[indices]) for name, indices in song_indices.items()
                if name in self.songs_left}
            

    def suggest(self, embeddings: pd.DataFrame):
        update_buffer = []
        self.t = 0

        self.songs_left = list(embeddings["name"].unique())

        while True:
            self.t += 1

            # To prevent model drift, we recommend n max songs from latest calculation, after which we update the model and recalc.

            song_values = self._calc_song_values(embeddings)
            for i in range(self.update_step):
                while True:
                    picked = max(song_values, key=song_values.get)

                    self.songs_left.remove(picked)
                    song_values.pop(picked)

                    song_data = embeddings[embeddings["name"] == picked]
                    if len(song_data.index) > 0:
                        break

                if i == 0 and self.t != 0:
                    picked = "(Updated model) " + picked

                user_score = yield picked.replace(".wav.csv", "")
                if user_score is None:
                    continue  # Gen seems to get called twice for some reason.

                update_buffer.extend([(chunk, user_score) for chunk in song_data.drop(columns=METADATA_COLS).to_numpy()])

            for chunk, r in update_buffer:
                if chunk is None:
                    print("Skip!")
                    continue

                self.exploration_matrix += np.outer(chunk, chunk)
                self.feature_coeffs += r * chunk
            update_buffer.clear()

            if len(self.songs_left) == 0:
                break

    def train(self, embeddings: pd.DataFrame, metrics: list[Metric], T: int) -> Trajectory:
        assert T > 0

        self.songs_left = list(embeddings["name"].unique())
        n_good_songs = len(embeddings[embeddings.score.isin(["liked", "loved"])]["name"].unique())

        trajectory = Trajectory(metrics={m.name: m(good_songs=n_good_songs) for m in metrics},
                                model_name=self.name, T=T)

        self.exploration_matrix = np.identity(self.n_dim)
        self.feature_coeffs = np.full(self.n_dim, self.feature_start)

        update_buffer = []

        for t in range(T):
            [m.calc(TrainEvent.STEP_START, self.exploration_matrix, self.feature_coeffs, 
                    self.alpha, embeddings) for m in trajectory.metrics.values()]

            song_values = self._calc_song_values(embeddings)
            picked = max(song_values, key=song_values.get)
            self.songs_left.remove(picked)
            
            song_data = embeddings[embeddings["name"] == picked]
            if len(song_data.index) == 0:
                continue
            reward = int(song_data.iloc[0]["score"] in["liked", "loved"])

            [m.calc(TrainEvent.REWARD_OBSERVED, reward) for m in trajectory.metrics.values()]
            
            update_buffer.extend([(chunk, reward) for chunk in song_data.drop(columns=METADATA_COLS).to_numpy()])
            n_good_songs -= reward

            if t > 0 and t % self.update_step == 0:
                for chunk, r in update_buffer:
                    self.exploration_matrix += np.outer(chunk, chunk)
                    self.feature_coeffs += r * chunk
                update_buffer.clear()
           
            [m.calc(TrainEvent.STEP_END, picked) for m in trajectory.metrics.values()]

            if len(self.songs_left) == 0 or n_good_songs == 0:
                trajectory.T = t
                self.songs_left = []
                break

        return trajectory

class Lin(Model):
    name = "Ridge (no UCB)"

    def __init__(self, n_dim: int, alpha: float, update_step: int, feature_start: float, description="") -> None:
        assert alpha > 0
        super().__init__()

        if description:
            self.name += f" ({description})"

        self.n_dim = n_dim

        if isinstance(feature_start, int):
            feature_start = float(feature_start)
        self.feature_start = feature_start

        self.alpha = alpha
        self.update_step = update_step

        self.exploration_matrix: np.ndarray  # A
        self.feature_coeffs: np.ndarray  # b

    def _calc_song_values(self, embeddings: pd.DataFrame, songs_left: list[str]) -> dict[str, float]:
        explor_inv = np.linalg.inv(self.exploration_matrix)  
        coeff = explor_inv @ self.feature_coeffs            
        
        all_chunks = embeddings.drop(columns=METADATA_COLS).to_numpy()
        song_indices = embeddings.groupby('name').indices
        predicted_rewards = all_chunks @ coeff         
        return {name: np.mean(predicted_rewards[indices]) for name, indices in song_indices.items()
                if name in songs_left}

    def train(self, embeddings: pd.DataFrame, metrics: list[Metric], T: int) -> Trajectory:
        assert T > 0

        songs_left = list(embeddings["name"].unique())
        n_good_songs = len(embeddings[embeddings.score.isin(["liked", "loved"])]["name"].unique())

        trajectory = Trajectory(metrics={m.name: m(good_songs=n_good_songs) for m in metrics},
                                model_name=self.name, T=T)

        self.exploration_matrix = np.identity(self.n_dim)
        self.feature_coeffs = np.full(self.n_dim, self.feature_start)

        update_buffer = []
        for t in range(T):
            [m.calc(TrainEvent.STEP_START, self.feature_coeffs, 
                    self.alpha, embeddings) for m in trajectory.metrics.values()]

            song_values = self._calc_song_values(embeddings, songs_left)
            picked = max(song_values, key=song_values.get)
            songs_left.remove(picked)
            
            song_data = embeddings[embeddings["name"] == picked]
            reward = int(song_data.iloc[0]["score"] in["liked", "loved"])

            [m.calc(TrainEvent.REWARD_OBSERVED, reward) for m in trajectory.metrics.values()]
            
            update_buffer.extend([(chunk, reward) for chunk in song_data.drop(columns=METADATA_COLS).to_numpy()])
            n_good_songs -= reward

            if t > 0 and t % self.update_step == 0:
                for chunk, r in update_buffer:
                    self.exploration_matrix += np.outer(chunk, chunk)
                    self.feature_coeffs += r * chunk
                update_buffer.clear()
           
            [m.calc(TrainEvent.STEP_END, picked) for m in trajectory.metrics.values()]

            if len(songs_left) == 0 or n_good_songs == 0:
                trajectory.T = t
                break

        return trajectory

from random import random, choice


class LinEps(Model):
    name = "Ridge (ε-greedy)"

    def __init__(self, n_dim: int, alpha: float, epsilon: float, update_step: int, feature_start: float, description="") -> None:
        assert alpha > 0
        assert 0 <= epsilon <= 1
        super().__init__()

        if description:
            self.name += f" ({description})"

        self.n_dim = n_dim

        if isinstance(feature_start, int):
            feature_start = float(feature_start)
        self.feature_start = feature_start

        self.alpha = alpha
        self.update_step = update_step
        self.epsilon = epsilon
        self.exploration_matrix: np.ndarray  # A
        self.feature_coeffs: np.ndarray  # b

    def _calc_song_values(self, embeddings: pd.DataFrame, songs_left: list[str]) -> dict[str, float]:
        explor_inv = np.linalg.inv(self.exploration_matrix)  
        coeff = explor_inv @ self.feature_coeffs            
        
        all_chunks = embeddings.drop(columns=METADATA_COLS).to_numpy()
        song_indices = embeddings.groupby('name').indices
        predicted_rewards = all_chunks @ coeff         
        return {name: np.mean(predicted_rewards[indices]) for name, indices in song_indices.items()
                if name in songs_left}

    def train(self, embeddings: pd.DataFrame, metrics: list[Metric], T: int) -> Trajectory:
        assert T > 0

        songs_left = list(embeddings["name"].unique())
        n_good_songs = len(embeddings[embeddings.score.isin(["liked", "loved"])]["name"].unique())

        trajectory = Trajectory(metrics={m.name: m(good_songs=n_good_songs) for m in metrics},
                                model_name=self.name, T=T)

        self.exploration_matrix = np.identity(self.n_dim)
        self.feature_coeffs = np.full(self.n_dim, self.feature_start)

        update_buffer = []
        for t in range(T):
            [m.calc(TrainEvent.STEP_START, self.feature_coeffs, 
                    self.alpha, embeddings) for m in trajectory.metrics.values()]

            if random() > self.epsilon:
                song_values = self._calc_song_values(embeddings, songs_left)
                picked = max(song_values, key=song_values.get)
            else:
                picked = choice(songs_left)

            songs_left.remove(picked)
            
            song_data = embeddings[embeddings["name"] == picked]
            if len(song_data.index) == 0:
                continue

            reward = int(song_data.iloc[0]["score"] in["liked", "loved"])

            [m.calc(TrainEvent.REWARD_OBSERVED, reward) for m in trajectory.metrics.values()]
            
            update_buffer.extend([(chunk, reward) for chunk in song_data.drop(columns=METADATA_COLS).to_numpy()])
            n_good_songs -= reward

            if t > 0 and t % self.update_step == 0:
                for chunk, r in update_buffer:
                    self.exploration_matrix += np.outer(chunk, chunk)
                    self.feature_coeffs += r * chunk
                update_buffer.clear()
           
            [m.calc(TrainEvent.STEP_END, picked) for m in trajectory.metrics.values()]

            if len(songs_left) == 0 or n_good_songs == 0:
                trajectory.T = t
                break

        return trajectory
