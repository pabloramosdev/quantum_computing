from numpy import ndarray
from numpy.random import random

from .optimization import Optimizer

from .qaoa_builder import QAOABuilder

from ..problems import (
    Problem, 
    QAOAProblemMapping
    )

class StatePreparation:

    def __init__(self, qaoa_config: dict, optimizer: Optimizer = Optimizer()):
        default_config = {"p": 1, "device_name": "default.qubit", "shots": 100, "init_params": None}

        overrides = qaoa_config or {}

        # check for invalid keys in overrides
        invalid = set(overrides) - set(default_config)
        if invalid:
            raise ValueError(f"Invalid QAOA config keys: {sorted(invalid)}. Allowed keys are: {sorted(default_config)}")

        # update default config with overrides
        default_config.update({k: overrides[k] for k in default_config if k in overrides})

        if type(default_config["p"]) is not int or default_config["p"] < 1:
            raise ValueError("p must be an int >= 1")
        if type(default_config["shots"]) is not int or default_config["shots"] <= 0:
            raise ValueError("shots must be a positive int")
        if type(default_config["device_name"]) is not str:
            raise ValueError("device_name must be a string")
        
        self.p = default_config["p"]
        self.device_name = default_config["device_name"]
        self.shots = default_config["shots"]
        init_params = default_config["init_params"]
        self.init_params = init_params if init_params is not None else random(2 * self.p)

        self.qaoa_builder = QAOABuilder(p=self.p, device_name=self.device_name, shots=self.shots)
        self.optimizer = optimizer

    def optimize(self, problem: Problem) -> ndarray:

        if not isinstance(problem, QAOAProblemMapping):
            raise TypeError("Problem must be an instance of QAOAProblemMapping")

        # Build cost QNode
        cost_qnode = self.qaoa_builder.build(problem, measurement="expval")
        
        return self.optimizer.optimize(cost_qnode, self.init_params)

    def counts(self, problem: Problem, params: ndarray = None) -> dict:

        if not isinstance(problem, QAOAProblemMapping):
            raise TypeError("Problem must be an instance of QAOAProblemMapping")
        if params is None:
            raise ValueError("params are required")
        if not isinstance(params, ndarray):
            raise TypeError("Parameters must be an ndarray")

        # Build cost QNode
        counts_qnode = self.qaoa_builder.build(problem, measurement="counts")

        # Evaluate counts with provided parameters
        counts = counts_qnode(params=params)

        return counts

    def correlations(self, problem: Problem, params: ndarray = None) -> list[tuple[float, tuple[int, int]]]:

        if not isinstance(problem, QAOAProblemMapping):
            raise TypeError("Problem must be an instance of QAOAProblemMapping")
        if params is None:
            raise ValueError("params are required")
        if not isinstance(params, ndarray):
            raise TypeError("Parameters must be an ndarray")

        # Build correlation QNode
        correlation_qnode = self.qaoa_builder.build(problem, measurement="correlations")

        # Evaluate correlations with provided parameters
        correlations = correlation_qnode(params=params)

        # Return correlation entries
        return StatePreparation._build_corr_entries(correlations, problem)

    @staticmethod
    def _build_corr_entries(results: ndarray, problem: Problem) -> list[tuple[float, tuple[int, int]]]:
        node_list = sorted(problem.nodes())
        edge_list = sorted((min(u, v), max(u, v)) for (u, v) in problem.edges())

        n = len(node_list)
        m = len(edge_list)
    
        if results.shape[0] != n + m:
            raise ValueError(f"Expected results length {n+m} (n={n}, m={m}), got {results.shape[0]}")

        one_point = results[:n]
        two_point = results[n:n + m]
        
        entries: list[tuple[float, tuple[int, int]]] = []

        # one-point: (node,node)
        for idx, node in enumerate(node_list):
            entries.append((float(one_point[idx]), (node, node)))

        # two-point: (u,v)
        for k, (u, v) in enumerate(edge_list):
            entries.append((float(two_point[k]), (u, v)))

        # Sort by absolute correlation value in descending order
        entries.sort(key=lambda t: abs(t[0]), reverse=True)
        return entries