from objects import Metric, TrainEvent

import numpy as np
import pandas as pd


class RidgeValues(Metric):
    name = "Ridge regression values"
    on_event = TrainEvent.STEP_START

    def calc_inner(self, 
                   exploration_matrix: np.ndarray, 
                   feature_coeffs: np.ndarray, 
                   alpha: float, 
                   embeddings: pd.DataFrame):
        pass

class UCBValues(Metric):
    name = "UCB values"
    on_event = TrainEvent.STEP_START

    def calc_inner(self, 
                   exploration_matrix: np.ndarray, 
                   feature_coeffs: np.ndarray, 
                   alpha: float, 
                   embeddings: pd.DataFrame):
        pass

class LinUCBValues(Metric):
    name = "Song values by components (regression, UCB, and combined)."
    on_event = TrainEvent.STEP_START
    children = [RidgeValues, UCBValues]

    def calc_inner(self, 
                   exploration_matrix: np.ndarray, 
                   feature_coeffs: np.ndarray, 
                   alpha: float, 
                   embeddings: pd.DataFrame):
        pass

class CoefficientChange(Metric):
    name = "Change in L2 Norm Coefficients"
    on_event = TrainEvent.STEP_START
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_theta = None

    def calc_inner(self, 
                   exploration_matrix: np.ndarray, 
                   feature_coeffs: np.ndarray, 
                   alpha: float, 
                   embeddings: pd.DataFrame):
        explor_inv = np.linalg.inv(exploration_matrix)
        coeff = explor_inv @ feature_coeffs

        if self.prev_theta is not None:
            self.values.append(np.mean(abs(coeff - self.prev_theta)))

        self.prev_theta = coeff

