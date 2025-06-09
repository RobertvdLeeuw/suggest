from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
import numpy as np


METADATA_COLS = ["artist", "song", "chunk", "orphan", "score", "filename", "name", "tags", "genres", "isrc"]


class TrainEvent(Enum):  # Used to define when to calc what metric.
    STEP_START = 0
    STEP_END = 1

    REWARD_OBSERVED = 2
    # SONG_VALUES_CALCULATED 
    # MODEL_UPDATED 
    # TRAINING_END 

# @dataclass
class Metric(ABC):
    name: str
    on_event: TrainEvent
    start_format: dict[str, list] | list = []  # dict when metric has subcomponents.

    def __init__(self, **kwargs):
        super().__init__()
        self.values = self.start_format

    def __iter__():
        """Important for plotting later on. Some metrics have multiple components, so we need to specify how to iterate over and overlay components onto same plot."""

        if self.start_format == []:
            return {"": self.values}

        if isinstance(self.start_format, dict) and all(isinstance(component, list) for component in self.start_format.values()):
            return self.values

        raise NotImplemented(f"Iter pattern not implement for object of structure: {self.start_format}")

    def calc(self, event: TrainEvent, *args, **kwargs):
        if event != self.on_event:
            return

        self.calc_inner(*args, **kwargs)
    
    @abstractmethod
    def calc_inner():
        pass

    def __repr__(self) -> str:
        return f"{self.name} on {self.on_event} initialized as {self.start_format}:\n\t{self.values}"  # TODO: Too large for human readable. Make something like Series.describe() with head() (head?).


@dataclass
class Trajectory: 
    model_name: str
    metrics: dict[str, Metric]
    T: int

    def __repr__(self) -> str:
        return f"Model: {self.model_name} (T={self.T}), Metrics:\n- {'\n- '.join([str(m) for m in self.metrics.values()])}"

class Model(ABC):
    name: str

    @abstractmethod
    def _calc_song_values(embeddings: pd.DataFrame, songs_left: list[str]) -> dict[str, float]:
        pass

    @abstractmethod
    def train(embeddings: pd.DataFrame, metrics: list[Metric], T: int) -> Trajectory:
        pass


