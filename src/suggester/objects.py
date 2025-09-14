from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
from operator import itemgetter
from functools import reduce
from copy import deepcopy

from jsonschema import validate
from jsonschema.exceptions import ValidationError

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
    Performance as PerformanceORM,
    ModelInstance as ModelInstanceORM,
)
from sqlalchemy import select, delete

import pandas as pd
import numpy as np


Json = dict | list

class TrainEvent(Enum):  # Used to define when to calc what metric.
    STEP_START = "step start"
    STEP_END = "step end"
    REWARD_OBSERVED = "reward observed"

class Metric(ABC):
    name: str
    id: str
    parent_id: str | None = None
    type: str | None = None  # Optional type field from schema

    children: list["Metric"] = None

    # TODO: Something np/polar object (f16)
    values: Json = None
    on_event: TrainEvent

    metric_schema: Json

    arguments: list[str]

    def __init__(self, parent_id=None):
        if self.children:
            for c in self.children:
                assert c.on_event == self.on_event, f"{self.name} triggers on event '{self.on_event}' but child '{c.name}' expects event '{c.on_event}'."
            assert hasattr(self, "id"), "Metric ID not assigned."
            self.children = [c(self.id) for c in self.children]
        else: 
            self.children = []

        self.values = []

    def calc(self, event: TrainEvent, **kwargs):
        if event != self.on_event:
            return

        for arg in self.arguments:
            assert arg in kwargs, f"About to calc {self.name}, but {arg} is missing from kwargs: " \
                                  f"'{"' '".join(kwargs.keys())}'."

        kwargs = {k: v for k, v in kwargs.items() if k in self.arguments}

        if self.children:
            for c in self.children:
                c.calc(event, **kwargs)
        else:
            self.calc_inner(**kwargs)
    
    @abstractmethod
    def calc_inner(self, **kwargs):
        pass

    async def upload(self):
        assert hasattr(self, "id"), "Metric ID not assigned."
        assert hasattr(self, "name"), "Metric name not assigned."

        async with get_session() as s:
            metric = MetricORM(
                metric_id=self.id,
                name=self.name,
                type=self.type,
                parent_id=self.parent_id,
                metric_schema=self.metric_schema
            )

            s.add(metric)
            await s.commit()

    async def upload_performance(self, trajectory_id: int, model_id: str, timestep: int, 
                               local_timestep: int = None, data: dict = None):
        """Upload performance data for this metric"""
        if data is None:
            data = self.values[-1] if self.values else {}
            
        async with get_session() as s:
            performance = PerformanceORM(
                metric_id=self.id,
                trajectory_id=trajectory_id,
                model_id=model_id,
                timestep=timestep,
                local_timestep=local_timestep,
                data=data
            )
            s.add(performance)
            await s.commit()

    def __str__(self): return repr(self)

    def __repr__(self) -> str:
        if isinstance(self.values, list) and self.values:
            if isinstance(self.values[0], (int, float)):
                body = pd.Series(self.values).describe()
            elif isinstance(self.values[0], dict):
                # Handle dict values - show keys and sample
                body = f"Dict metrics with keys: {list(self.values[0].keys()) if self.values else []}"
            else:
                body = f"Values: {len(self.values)} entries"
        else: 
            body = "No values recorded"

        return f"{self.name} on {self.on_event}:\n\t{body}"

@dataclass
class Trajectory: 
    model_name: str
    model_id: str
    metrics: dict[str, Metric]
    T: int
    user_id: int
    funnel_id: int | None = None
    
    trajectory_id: int = None

    def __str__(self): return repr(self)

    def __repr__(self) -> str:
        metric_names = [m.name for m in self.metrics.values()]
        return f"""
====== {self.model_name} (Trajectory) =====
  Metrics:
    - {"\n\t- ".join(metric_names)}
        """

    async def upload(self):
        async with get_session() as s:
            trajectory = TrajectoryORM(
                user_id=self.user_id,
                funnel_id=self.funnel_id,
                started=datetime.utcnow(),  # TODO: Better time tracking for PROD use.
                ended=None, 
                timesteps=self.T,
                on_history=False  # TODO: Expose param 
            )
            s.add(trajectory)
            await s.flush()  # Get the ID
            self.trajectory_id = trajectory.trajectory_id
            
            for metric_name, metric in self.metrics.items():
                await metric.upload()
            
            await s.commit()

def validated(obj: Json, schema: Json, context: str = "object") -> Json:
    try:
        validate(instance=obj, schema=schema)
        return obj
    except ValidationError as e:
        raise ValueError(f"Schema validation failed for {context}: {e}") from e

class Model(ABC):
    name: str
    id: str  # model_id
    allowed_metrics: list[Metric]

    param_schema: Json
    hyperparam_schema: Json

    def __init__(self, metrics: list[Metric], n_out: int, n_in: int = 0):
        assert n_in >= 0, "Model must accept more than 1 item in, or set to 0 for any number in."
        assert n_out > 0, "Model must allow at least 1 item out."
        assert n_in > n_out or n_in == 0, "Model should filter (in > out)."

        for m in metrics:
            assert any(isinstance(m, allowed_type) for allowed_type in self.allowed_metrics), \
                f"Unallowed metric for {self.name}: {m.name}."

        self.n_in = n_in
        self.n_out = n_out
        self.metrics = [m() if isinstance(m, type) else m for m in metrics]

    def __str__(self): return repr(self)

    def __repr__(self) -> str:
        m = ", \n".join([m.name for m in self.metrics])
        return f"{self.name} ({self.n_in} -> {self.n_out}), metrics: {m}"
        
    def calc_metrics(self, event: TrainEvent, **kwargs):
        [m.calc(event, **kwargs) for m in self.metrics]

    def process(self, items: np.ndarray) -> np.ndarray:
        for col in ["song_id", "chunk_id", "embedding"]:
            assert col in items.dtype.names, f"{col} missing in embeddings in."

        assert len(np.unique(items["song_id"])) == self.n_in or self.n_in == 0, \
            f"{self.name} expected {self.n_in} songs in, got {len(np.unique(items['song_id']))}."

        items_out = self.process_inner(items)

        for col in ["song_id", "chunk_id", "embedding"]:
            assert col in items_out.dtype.names, f"{col} missing in embeddings out."

        assert len(np.unique(items_out["song_id"])) == self.n_out, \
            f"{self.name} expected {self.n_out} songs out, got {len(np.unique(items_out['song_id']))}."

        return items_out

    @abstractmethod
    def process_inner(self, items: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def load(cls, hyperparams, params):
        raise NotImplementedError
    
    @abstractmethod
    def serialize_params(self) -> dict:
        """
        Return current model parameters as JSON-serializable dict
        """
        pass
    
    @abstractmethod
    def serialize_hyperparams(self) -> dict:
        """
        Return current hyperparameter values as JSON-serializable dict
        """
        pass
    
    async def upload(self):
        assert hasattr(self, "id"), "Model ID not assigned."
        assert hasattr(self, "name"), "Model name not assigned."
        assert hasattr(self, "param_schema"), "Model parameter schema not assigned."
        assert hasattr(self, "hyperparam_schema"), "Model hyperparameter schema not assigned."

        async with get_session() as s:
            model = ModelORM(
                model_id=self.id,
                model_name=self.name,
                param_schema=self.param_schema,
                hyperparam_schema=self.hyperparam_schema
            )
            s.add(model)
            
            # Upload allowed metrics for this model
            for metric_class in self.allowed_metrics:
                # Create instance to get metric info
                if isinstance(metric_class, type):
                    metric_instance = metric_class()
                else:
                    metric_instance = metric_class
                    
                await metric_instance.upload()
                
                # Create model-metric relationship
                from models import ModelMetric
                model_metric = ModelMetric(
                    model_id=self.id,
                    metric_id=metric_instance.id
                )
                s.add(model_metric)
            
            await s.commit()
    
    async def upload_instance(self, trajectory_id: int):
        """Upload model instance (params + hyperparams) for a specific trajectory"""
        async with get_session() as s:
            model_instance = ModelInstanceORM(
                model_id=self.id,
                trajectory_id=trajectory_id,
                params=validated(self.serialize_params(), 
                                 self.param_schema, 
                                 "model parameters"),
                hyperparams=validated(self.serialize_hyperparams(),
                                      self.hyperparam_schema, 
                                      "model hyperparameters")
            )
            s.add(model_instance)
            await s.commit()

    async def upload_performance(self, trajectory_id: int, timestep: int, local_timestep: int = None):
        """Upload performance for all metrics of this model"""
        for metric in self.metrics:
            await metric.upload_performance(
                trajectory_id=trajectory_id,
                model_id=self.id,
                timestep=timestep,
                local_timestep=local_timestep
            )

    def train(self, songs: any, T: int) -> Trajectory: 
        raise NotImplementedError

class Funnel:
    def __init__(self, name: str, models: list[Model]):
        self.name = name
        self.models = models
        self.funnel_id = None
        self._model_layers_match()

    def _model_layers_match(self):   
        assert self.models[0].n_in == 0, f"First layer of {self.name} needs to accept any n items (n_in should be 0)."

        for i in range(len(self.models)-1):
            n_out = self.models[i].n_out
            n_in = self.models[i+1].n_in

            assert n_out == n_in, f"Mismatching layer in/out between models {i+1}-{i+2}: {n_out} -> {n_in}."

    def __str__(self): return repr(self)
    
    def __repr__(self) -> str:
        model_strs = [str(m) for m in self.models]
        return f"""
====== {self.name} (Funnel) =====
  Models:
    - {"\n\t- ".join(model_strs)}
        """

    def suggest(self, items: np.ndarray) -> np.ndarray:
        return reduce(lambda x, model: model.process(x), self.models, items)

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
                    model_id=model.id,
                    position=position
                )
                s.add(funnel_model)
            
            await s.commit()

    async def train(self, songs: any, T: int, user_id: int) -> Trajectory: 
        """
        Train the funnel and return a trajectory with all model performances
        """
        # Create trajectory
        # For simplicity, using the first model's metrics - in practice you might want funnel-level metrics
        trajectory = Trajectory(
            model_name=self.name,
            model_id=f"funnel_{self.funnel_id}",
            metrics={},  # Aggregate from all models or define funnel-specific metrics
            T=T,
            user_id=user_id,
            funnel_id=self.funnel_id
        )
        
        await trajectory.upload()
        
        # Upload model instances for all models in the funnel
        for model in self.models:
            await model.upload_instance(trajectory.trajectory_id)
        
        return trajectory
