from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from functools import reduce

from copy import deepcopy

import pandas as pd
import numpy as np


class Funnel:
    # TODO: Do we need funnel-level metrics as well or just at layer (singular model) level?

    def __init__(self, name: str, id: str, layers: list[Layer]):
        self.name = name

        self.layers = layers
        assert self.layer_n_match()

    def layer_n_match():   
        assert self.layers[0].n_in == 0, f"First layer of {self.name} needs to accept any n items (n_in should be 0)."

        for i in range(len(self.layers)-1):
            n_out = self.layers[i].n_out
            n_in = self.layers[i+1].n_in

            if n_out != n_in:
                raise Exception(
                    f"Mismatching layer in/out between layers {i+1}-{i+2}: {n_out} -> {n_in}"
                )

        return True

    def __str__(self): return repr(self)
    
    def __repr__(self) -> str:
        return f"""
====== {self.name} (Funnel) =====
  Models:
    - {"\n\t- ".join(self.layers)}
        """

    def suggest(self, items: np.ndarray) -> list[str]:
        return reduce(lambda x, layer: layer.process(x), self.layers)

    @abstractmethod
    def train(songs: any, T: int) -> Trajectory: pass

Json = dict | list

class Model(ABC):
    name: str
    allowed_metrics: list[Metric]

    param_schema: Json

    def __init__(self, metrics: list[Metric], n_out: int, n_in: int = 0):
        assert n_in >= 0, "Layer must accept more than 1 item in, or set to 0 for any number in."
        assert n_out > 0, "Layer must allow at least 1 item out."
        assert n_in > n_out, "Layer should filter (in > out)."

        for m in metrics:
            assert all(m in self.allowed_metrics), f"Unallowed metric for {self.name}: {m.name}."

        self.n_in = n_in
        self.n_out = n_out
        self.metrics = metrics

    def __str__(self): return repr(self)

    def __repr__(self) -> str:
        m = ", ".join([m.name for m in self.matrics])
        return f"{self.name} ({self.n_in} -> {self.n_out}), metrics: {m}"
        
    def process(self, embeddings: np.ndarray, song_ids: list) -> tuple[np.ndarray, list]:
        assert embeddings.shape[0] == len(song_ids), f"{self.name} got {embeddings.shape[0]} embeddings, but {len(song_ids} song ids as input."
        assert embeddings.shape[0] == self.n_in, f"{self.name} expected {self.n_in} items in, got {items.shape[0]}."

        embeddings_out, ids_out = self.process_inner(embeddings, song_ids)

        assert embeddings_out.shape[0] == self.n_out, f"{self.name} expected {self.n_out} items out, got {out.shape[0]}."
        assert embeddings_out.shape[0] == len(ids_out), f"{self.name} got {embeddings_out.shape[0]} embeddings, but {len(ids_out} song ids as output."
        return embeddings_out

    @abstractmethod
    def process_inner(self, items: np.ndarray, song_ids: list) -> tuple[np.ndarray, list]:
        pass

    @abstractmethod
    @classmethod
    def load(cls, hyperparams, params): 
        

    @abstractmethod
    def to_db(self): pass

    @abstractmethod
    def validate_params(self): pass


@dataclass
class Trajectory: 
    model_name: str
    model_id: str
    metrics: dict[str, Metric]
    T: int

    def __str__(self): return repr(self)

    def __repr__(self) -> str:
        return f"""
====== {self.model_name} (Trajectory) =====
  Metrics:
    - {"\n\t- ".join([str(m) for m in self.metrics])}
        """

    @abstractmethod
    def to_db(self): pass

# ======================== OLD ===================

class TrainEvent(Enum):  # Used to define when to calc what metric.
    STEP_START = "step start"
    STEP_END = "step end"
    REWARD_OBSERVED = "reward observed"

class MetricType(Enum):  # So we can group different metrics into same plot (e.g. expected values for different models).
    PREDICTED_VALUE = "predicted value"
    MODEL_PARAM_CHANGE = "model param change"

class Metric(ABC):
    name: str
    id: str

    type: MetricType
    children: list["Metric"]

    # TODO: Something np/polar object (f16)
    values: list[dict[str, float | int] | list[float | int]] = []
    on_event: TrainEvent

    def __init__(self, **kwargs):
        for c in self.children:
            assert c.on_event != self.on_event, f"{self.name} triggers on event '{self.on_event}' but child '{c.name}' expects event '{c.on_event}'."
            assert c.type != self.type, f"{self.name} is of type '{self.type}' but child '{c.type}' is of type '{c.type}'."

    def calc(self, event: TrainEvent, *args, **kwargs):
        if event != self.on_event:
            return

        if self.children:
            [c.calc(event, *args, **kwargs) for c in self.children]

        # TODO: Should only children calc, and higher level values calced on the fly for plots? 
            # Less data in DB this way.
        self.calc_inner(*args, **kwargs)
    
    @abstractmethod
    def calc_inner():
        pass

    def __str__(self): return repr(self)

    def __repr__(self) -> str:
        if self.start_format == []:
            body = pd.Series(self.values).describe()
        elif isinstance(self.start_format, dict) and all(isinstance(component, list) for component in self.start_format.values()):
            body = {k: pd.Series(v).describe() for k, v in self.values.items()}
        else: 
            raise NotImplementedError(f"Repr pattern not implement for object of structure: {self.start_format}")

        return f"{self.name} on {self.on_event} initialized as {self.start_format}:\n\t{body}"


