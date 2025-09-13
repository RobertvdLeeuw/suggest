from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
from operator import itemgetter
from functools import reduce
from copy import deepcopy

import traceback
import logging
LOGGER = logging.getLogger(__name__)

from db import get_session
from models import (
    EmbeddingJukeMIR, EmbeddingAuditus, 
    Funnel as FunnelORM,
    Metric as MetricORM,
    Trajectory as TrajectoryORM,
    Model as ModelORM,
    FunnelModel as FunnelModelORM,
    ModelPerformance as ModelPerformanceORM,
    ParamInstance as ParamInstanceORM,
    Hyperparameter as HyperparameterORM,
    HPInstance as HPInstanceORM
)
from sqlalchemy import select, delete

import pandas as pd
import numpy as np


Json = dict | list

class TrainEvent(Enum):  # Used to define when to calc what metric.
    STEP_START = "step start"
    STEP_END = "step end"
    REWARD_OBSERVED = "reward observed"

# class MetricType(Enum):  # So we can group different metrics into same plot (e.g. expected values for different models).
#     PREDICTED_VALUE = "predicted value"
#     MODEL_PARAM_CHANGE = "model param change"

class Metric(ABC):
    name: str
    id: str
    parent_id: str | None = None

    # type: MetricType
    children: list["Metric"] = None

    # TODO: Something np/polar object (f16)
    values: list[dict[str, float | int] | list[float | int]] = None
    on_event: TrainEvent

    arguments: list[str]

    def __init__(self, parent_id=None):
        if self.children:
            for c in self.children:
                assert c.on_event == self.on_event, f"{self.name} triggers on event '{self.on_event}' but child '{c.name}' expects event '{c.on_event}'."
                # assert c.type == self.type, f"{self.name} is of type '{self.type}' but child '{c.type}' is of type '{c.type}'."
            assert hasattr(self, "id"), "Model ID not assigned."
            self.children = [c(self.id) for c in self.children]
        else: 
            self.children = []

        self.values = []

    def calc(self, event: TrainEvent, **kwargs):
        if event != self.on_event:
            return

        for arg in self.arguments:
            assert arg in kwargs, f"About to calc {name}, but {arg} is missing from kwargs: " \
                                  f"'{"' ".join(kwargs)}'."

        kwargs = {k: v for  k, v in kwargs.items() if k in self.arguments}

        if self.children:
            for c in self.children:
                c.calc(event, **kwargs)
        else:
            self.calc_inner(**kwargs)
    
    @abstractmethod
    def calc_inner():
        pass

    async def upload(self):
        assert hasattr(self, "id"), "Model ID not assigned."
        assert hasattr(self, "name"), "Model name not assigned."

        async with get_session() as s:
            metric = MetricORM(metric_id=self.id,
                               name=self.name,
                               type=None,
                               parent_id=self.parent_id)

            s.add(metric)
            s.commit()

    def __str__(self): return repr(self)

    def __repr__(self) -> str:
        if isinstance(self.values, list):
            body = pd.Series(self.values).describe()
            # TODO: Update other cases below.
        elif isinstance(self.start_format, dict) and all(isinstance(component, list) for component in self.start_format.values()):
            body = {k: pd.Series(v).describe() for k, v in self.values.items()}
        else: 
            raise NotImplementedError(f"Repr pattern not implement for object of structure: {self.start_format}")

        return f"{self.name} on {self.on_event}:\n\t{body}"

@dataclass
class Trajectory: 
    model_name: str
    model_id: str
    metrics: dict[str, Metric]
    T: int
    
    trajectory_id: int = None

    def __str__(self): return repr(self)

    def __repr__(self) -> str:
        return f"""
====== {self.model_name} (Trajectory) =====
  Metrics:
    - {"\n\t- ".join([str(m) for m in self.metrics])}
        """

     async def upload(self):
        async with get_session() as s:
            trajectory = TrajectoryORM(
                model_id=self.model_id,
                started=datetime.utcnow(),  # TODO: Better time tracking for PROD use.
                ended=None, 
                timesteps=self.T,
                on_history=False  # TODO: Expose param 
            )
            s.add(trajectory)
            await s.flush()  # Get the ID
            self.trajectory_id = trajectory.trajectory_id
            
            for metric_name, metric in self.metrics.items():
                metric.trajectory_id = self.trajectory_id
                await metric.upload()
            
            await s.commit()


class Model(ABC):
    name: str
    allowed_metrics: list[Metric]

    param_schema: Json

    def __init__(self, metrics: list[Metric], n_out: int, n_in: int = 0):
        assert n_in >= 0, "Model must accept more than 1 item in, or set to 0 for any number in."
        assert n_out > 0, "Model must allow at least 1 item out."
        assert n_in > n_out or n_in == 0, "Model should filter (in > out)."

        for m in metrics:
            assert all(m in self.allowed_metrics for m in metrics), f"Unallowed metric for {self.name}: {m.name}."

        self.n_in = n_in
        self.n_out = n_out
        self.metrics = [m() for m in metrics]

    def __str__(self): return repr(self)

    def __repr__(self) -> str:
        m = ", \n".join([m.name for m in self.metrics])
        return f"{self.name} ({self.n_in} -> {self.n_out}), metrics: {m}"
        
    def calc_metrics(self, event: TrainEvent, **kwargs):
        [m.calc(event, **kwargs) for m in self.metrics]

    def process(self, items: np.ndarray) -> np.ndarray:
        for col in ["song_id", "chunk_id", "embedding"]:
            assert col in items.dtype.names, f"{col} missing in embeddings in."

        assert len(np.unique(items["song_id"])) == self.n_in or self.n_in == 0, f"{self.name} expected {self.n_in} songs in, got {len(np.unique(items["song_id"]))}."

        items_out = self.process_inner(items)

        for col in ["song_id", "chunk_id", "embedding"]:
            assert col in items_out.dtype.names, f"{col} missing in embeddings out."

        assert len(np.unique(items_out["song_id"])) == self.n_out, f"{self.name} expected {self.n_out} songs out, got {len(np.unique(items_out["song_id"]))}."

        return items_out

    @abstractmethod
    def process_inner(self, items: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def load(cls, hyperparams, params):
        raise NotImplementedError

  @abstractmethod
    def hyperparams_serialize(self) -> dict[int, dict]:
        """
        Return hyperparameters as dict with hp_id as key.
        Each value should be: {'type': HyperparameterType, 'min': float, 'max': float}
        """
        pass
    
    @abstractmethod
    def params_serialize(self) -> dict:
        """
        Return current model parameters as JSON-serializable dict
        """
        pass
    
    @abstractmethod
    def hyperparams_instance_serialize(self) -> dict[int, float]:
        """
        Return current hyperparameter values as dict with hp_id -> value
        """
        pass
    
    async def upload(self):
        async with get_session() as s:
            model = ModelORM(
                model_id=self.model_id if hasattr(self, 'model_id') else self.name,
                model_name=self.name,
                param_schema=self.param_schema
            )
            s.add(model)
            
            # Upload hyperparameters using the serialize method
            hyperparams = self.hyperparams_serialize()
            for hp_id, hp_config in hyperparams.items():
                hyperparameter = HyperparameterORM(
                    model_id=model.model_id,
                    hp_id=hp_id,
                    type=hp_config['type'],
                    min=hp_config.get('min'),
                    max=hp_config.get('max')
                )
                s.add(hyperparameter)
            
            await s.commit()
    
    async def upload_params(self, trajectory_id: int):
        """Upload parameter instance for a specific trajectory"""
        async with get_session() as s:
            params = self.params_serialize()
            param_instance = ParamInstanceORM(
                model_id=self.model_id if hasattr(self, 'model_id') else self.name,
                trajectory_id=trajectory_id,
                params=params
            )
            s.add(param_instance)
            await s.commit()
    
    async def upload_hp_instance(self, trajectory_id: int):
        """Upload hyperparameter values for a specific trajectory"""
        async with get_session() as s:
            hyperparams = self.hyperparams_serialize()
            hp_values = self.hyperparams_instance_serialize()
            
            for hp_id, hp_value in hp_values.items():
                hp_type = hyperparams[hp_id]['type']
                
                hp_instance = HPInstanceORM(
                    model_id=self.model_id if hasattr(self, 'model_id') else self.name,
                    trajectory_id=trajectory_id,
                    hp_id=hp_id,
                    type=hp_type,
                    value=float(hp_value)
                )
                s.add(hp_instance)
            await s.commit()
    
    async def upload_params(self, trajectory_id: int, params: dict):
        """Upload parameter instance for a specific trajectory"""
        async with get_session() as s:
            param_instance = ParamInstanceORM(
                model_id=self.model_id if hasattr(self, 'model_id') else self.name,
                trajectory_id=trajectory_id,
                params=params
            )
            s.add(param_instance)
            await s.commit()
    
    async def upload_hp_instance(self, trajectory_id: int, hyperparameters: dict):
        """Upload hyperparameter values for a specific trajectory"""
        async with get_session() as s:
            for hp_id, hp_value in hyperparameters.items():
                # You'll need to determine the type based on your hp schema
                hp_type = self._get_hp_type(hp_id)  # Implement this method
                
                hp_instance = HPInstanceORM(
                    model_id=self.model_id if hasattr(self, 'model_id') else self.name,
                    trajectory_id=trajectory_id,
                    hp_id=hp_id,
                    type=hp_type,
                    value=float(hp_value)  # Convert to float as per schema
                )
                s.add(hp_instance)
            await s.commit()

    # @abstractmethod
    def validate_params(self):
        pass

    def train(songs: any, T: int) -> Trajectory: 
        raise NotImplementedError

class Funnel:
    # TODO: Do we need funnel-level metrics as well or just at layer (singular model) level?

    def __init__(self, name: str, id: str, layers: list[Model]):
        self.name = name

        self.models = models
        self.model_layers_match()

    def model_layers_match():   
        assert self.models[0].n_in == 0, f"First layer of {self.name} needs to accept any n items (n_in should be 0)."

        for i in range(len(self.models)-1):
            n_out = self.models[i].n_out
            n_in = self.models[i+1].n_in

            assert n_out == n_in, f"Mismatching layer in/out between models {i+1}-{i+2}: {n_out} -> {n_in}."

    def __str__(self): return repr(self)
    
    def __repr__(self) -> str:
        return f"""
====== {self.name} (Funnel) =====
  Models:
    - {"\n\t- ".join([str(m) for m in self.models])}
        """

    def suggest(self, items: np.ndarray, song_ids: list[str]) -> list[str]:
        # TODO: This is more complex when latter part of funnel ran run multiple times.
        return reduce(lambda x, model: model.process(*x), self.models, items)

    async def upload(self):
        async with get_session() as s:
            # Create funnel record
            funnel = FunnelORM(
                funnel_name=self.name
            )
            s.add(funnel)
            await s.flush()  # Get the ID
            self.funnel_id = funnel.funnel_id
            
            # Upload all models first
            for model in self.models:
                await model.upload()
            
            # Create funnel-model relationships
            for position, model in enumerate(self.models):
                funnel_model = FunnelModelORM(
                    funnel_id=self.funnel_id,
                    model_id=model.model_id if hasattr(model, 'model_id') else model.name,
                    position=position
                )
                s.add(funnel_model)
            
            await s.commit()

    def train(songs: any, T: int) -> Trajectory: 
        raise NotImplementedError


