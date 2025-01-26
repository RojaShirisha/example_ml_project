from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import json
import pathlib
import pickle
from typing import Any, Dict, Tuple

import pandas


class BasePredictor(ABC):
    """
    An abstract predictor baseclass.

    Attrs:
        group: type of predictor (e.g. algorithm name)
        params: a dataclass with parameters
        model: a model object with trained weights
    """

    group: str
    params: "dataclass"
    model: Any

    @classmethod
    def load(cls, path: str) -> Tuple[Dict[str, Any], Any]:
        """Load predictor from file"""
        path = pathlib.Path(path)
        params = json.load(open(path / "params.json", "r"))
        model = pickle.load(open(path / "model.pkl", "rb"))
        predictor = cls(params)
        predictor.model = model
        return predictor

    def save(self, path: str):
        """Save predictor to disk.

        Args:
            path: directory to save all predictor files in
        """
        path = pathlib.Path(path)
        json.dump(asdict(self.params), open(path / "params.json", "w"))
        pickle.dump(self.model, open(path / "model.pkl", "wb"))

    @abstractmethod
    def fit(self, x_data: Any, y_data: Any):
        pass

    @abstractmethod
    def predict(self, x_data: Any) -> Any:
        pass


@dataclass
class SVMParameters:
    peptide_lengths: Tuple[int, ...]


class SVMPredictor(BasePredictor):
    group = "svm"
    params = SVMParameters
    model = None

    def __init__(self, params: Dict[str, Any]):
        self.params = SVMParameters(**params)

    def fit(self, x_data: pandas.DataFrame, y_data: pandas.Series):
        print("Imagine there's some training happening here...")
        self.model = 42

    def predict(self, x_data: pandas.DataFrame) -> pandas.DataFrame:
        """Always predicts 42"""
        valid_rows = x_data["peptide"].str.len().isin(self.params.peptide_lengths)
        x_data["score"] = None
        x_data.loc[valid_rows, "score"] = self.model
        return x_data


@dataclass
class NeuralParameters:
    peptide_lengths: Tuple[int, ...]


class NeuralPredictor(BasePredictor):

    group = "neural_network"
    params = NeuralParameters
    model = None

    def __init__(self, params: Dict[str, Any]):
        self.params = NeuralParameters(**params)

    def fit(self, x_data: pandas.DataFrame, y_data: pandas.Series):
        print("Imagine there's some training happening here...")
        self.model = 57

    def predict(self, x_data) -> pandas.DataFrame:
        """Always predicts 57"""
        valid_rows = x_data["peptide"].str.len().isin(self.params.peptide_lengths)
        x_data["score"] = None
        x_data.loc[valid_rows, "score"] = self.model
        return x_data
