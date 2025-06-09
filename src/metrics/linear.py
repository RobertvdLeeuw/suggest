from objects import Metric, TrainEvent, METADATA_COLS

import numpy as np
import pandas as pd


class SongValuesUCB(Metric):
    name = "Song values by components (regression, UCB, and combined)."
    on_event = TrainEvent.STEP_START
    start_format = {"Base Reward": [],
                    "UCB Component": [],
                    "Total": []}

    def calc_inner(self, 
                   exploration_matrix: np.ndarray, 
                   feature_coeffs: np.ndarray, 
                   alpha: float, 
                   embeddings: pd.DataFrame):
        explor_inv = np.linalg.inv(exploration_matrix)
        coeff = explor_inv @ feature_coeffs

        all_chunks = embeddings.drop(columns=METADATA_COLS).to_numpy()
        song_indices = embeddings.groupby('name').indices

        uncertainty_terms = alpha * np.sqrt(np.sum(all_chunks @ explor_inv * all_chunks, axis=1))
        predicted_rewards = all_chunks @ coeff
        ucb_values = predicted_rewards + uncertainty_terms

        self.values["Base Reward"].append({name: np.mean(predicted_rewards[indices]) for name, indices in song_indices.items()})
        self.values["UCB Component"].append({name: np.mean(uncertainty_terms[indices]) for name, indices in song_indices.items()})
        self.values["Total"].append({name: np.mean(ucb_values[indices]) for name, indices in song_indices.items()})
       
# TODO: How can we now design an offline learning or linTS model with different metric but applied to same subplot? Some extra attribute?
