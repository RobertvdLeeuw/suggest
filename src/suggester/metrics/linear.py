from suggester.objects import Metric, TrainEvent

import numpy as np
import pandas as pd


class RidgeValues(Metric):
    name = "Ridge regression values"
    id = "ridge_values"
    on_event = TrainEvent.STEP_START
    
    metric_schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "required": ["values"]
    }

    arguments = ["exploration_matrix", "feature_coeffs", "items"]

    def calc_inner(self, 
                   exploration_matrix: np.ndarray, 
                   feature_coeffs: np.ndarray, 
                   items: np.ndarray):
        explor_inv = np.linalg.inv(exploration_matrix)
        coeff = explor_inv @ feature_coeffs  # Theta 
        predicted_rewards = items["embedding"] @ coeff
        
        # Store the predicted values for each item
        self.values.append({
            "values": predicted_rewards.tolist(),
            "song_ids": items["song_id"].tolist()
        })


class UCBValues(Metric):
    name = "UCB values"
    id = "ucb_values"
    on_event = TrainEvent.STEP_START
    
    metric_schema = {
        "type": "object", 
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "number"}
            },
            "alpha": {"type": "number"}
        },
        "required": ["values", "alpha"]
    }

    arguments = ["exploration_matrix", "feature_coeffs", "items", "alpha"]

    def calc_inner(self, 
                   exploration_matrix: np.ndarray, 
                   feature_coeffs: np.ndarray, 
                   items: np.ndarray,
                   alpha: float):
        explor_inv = np.linalg.inv(exploration_matrix)
        uncertainty_terms = np.sum(items["embedding"] @ explor_inv * items["embedding"], axis=1)
        ucb_bonus = alpha * np.sqrt(uncertainty_terms)
        
        self.values.append({
            "values": ucb_bonus.tolist(),
            "alpha": alpha,
            "song_ids": items["song_id"].tolist()
        })


class LinUCBValues(Metric):
    name = "Song values by components (regression, UCB, and combined)."
    id = "linucb_values"
    on_event = TrainEvent.STEP_START
    children = [RidgeValues, UCBValues]
    
    metric_schema = {
        "type": "object",
        "properties": {
            "combined_values": {
                "type": "array", 
                "items": {"type": "number"}
            },
            "top_songs": {
                "type": "array",
                "items": {"type": "integer"}
            }
        },
        "required": ["combined_values", "top_songs"]
    }

    arguments = ["exploration_matrix", "feature_coeffs", "items", "alpha"]

    def calc_inner(self, 
                   exploration_matrix: np.ndarray, 
                   feature_coeffs: np.ndarray, 
                   items: np.ndarray,
                   alpha: float):
        explor_inv = np.linalg.inv(exploration_matrix)
        coeff = explor_inv @ feature_coeffs
        
        uncertainty_terms = np.sum(items["embedding"] @ explor_inv * items["embedding"], axis=1)
        predicted_rewards = items["embedding"] @ coeff
        ucb_values = predicted_rewards + alpha * np.sqrt(uncertainty_terms)
        
        # Get top songs by UCB value
        unique_songs, inverse_indices, counts = np.unique(
            items["song_id"], return_inverse=True, return_counts=True
        )
        song_ucb_avgs = np.bincount(inverse_indices, weights=ucb_values) / counts
        top_indices = np.argsort(song_ucb_avgs)[-10:]  # Top 10 songs
        
        self.values.append({
            "combined_values": ucb_values.tolist(),
            "top_songs": unique_songs[top_indices].tolist(),
            "song_ids": items["song_id"].tolist()
        })


class CoefficientChange(Metric):
    name = "Change in L2 Norm Coefficients"
    id = "coefficient_change"
    on_event = TrainEvent.STEP_START
    
    metric_schema = {
        "type": "object",
        "properties": {
            "l2_change": {"type": "number"},
            "step": {"type": "integer"}
        },
        "required": ["l2_change", "step"]
    }

    arguments = ["exploration_matrix", "feature_coeffs", "items"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_theta = None
        self.step = 0

    def calc_inner(self, 
                   exploration_matrix: np.ndarray, 
                   feature_coeffs: np.ndarray, 
                   items: np.ndarray):
        explor_inv = np.linalg.inv(exploration_matrix)
        coeff = explor_inv @ feature_coeffs

        if self.prev_theta is not None:
            l2_change = float(np.linalg.norm(coeff - self.prev_theta))
            self.values.append({
                "l2_change": l2_change,
                "step": self.step
            })

        self.prev_theta = coeff.copy()
        self.step += 1
