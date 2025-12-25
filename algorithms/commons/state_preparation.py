from dataclasses import dataclass, field
from typing import Optional

from numpy import ndarray
from numpy.random import random
from networkx import Graph
from scipy.optimize import minimize

from pennylane import device

from .qaoa_builder import QAOABuilder

from ..problems import (
    Problem, 
    QAOAProblemMapping
    )


@dataclass
class QAOAConfig:
    p: int = 1
    device_name: str = "default.qubit"
    shots: int = 100
    init_params: Optional[ndarray] = field(default=None)

    def __post_init__(self):
        if self.p < 1:
            raise ValueError("p must be >= 1")
        if self.init_params is None:
            self.init_params = random(2 * self.p)

@dataclass
class OptimizationConfig:
    method: str = "COBYLA"
    tol: float = 1e-2
    maxiter: int = 100

class StatePreparation:

    def __init__(self, qaoa_config: QAOAConfig, optimization_config: OptimizationConfig = OptimizationConfig()):
        """Constructor for StatePreparation.
        Args:
            qaoa_config (QAOAConfig): Configuration for QAOA.
            optimization_config (OptimizationConfig): Configuration for optimization.
        """
        self.qaoa_config = qaoa_config
        self.optimization_config = optimization_config

    def prepare_qaoa_state(self, problem: Problem) -> ndarray:

        if not isinstance(problem, QAOAProblemMapping):
            raise TypeError("Problem must be an instance of QAOAProblemMapping")
        
        device_name = self.qaoa_config.device_name
        p = self.qaoa_config.p
        shots = self.qaoa_config.shots

        method = self.optimization_config.method
        tol = self.optimization_config.tol
        maxiter = self.optimization_config.maxiter
        init_params = self.qaoa_config.init_params

        # Build cost hamiltonian and mixer hamiltonian
        cost_h, mixer_h = problem.cost_and_mixer_hamiltonians()
        
        # Build QAOA circuit
        qaoa_circuit = QAOABuilder(p=p, cost_h=cost_h, mixer_h=mixer_h)

        # Set up device
        graph = problem.get_graph()
        wires = list(graph.nodes())
        dev = device(name=device_name, wires=wires, shots=shots)

        # Build cost QNode
        cost_qnode = qaoa_circuit.build_qnode(dev)

        # Optimize parameters
        results = minimize(cost_qnode, x0=init_params, method=method, tol=tol, options={"maxiter": maxiter})
        
        if not results.success:
            raise RuntimeError("Optimization failed: " + results.message)
        
        return results.x

    def get_counts(self, problem: Problem, params: ndarray = None) -> dict:

        if not isinstance(problem, QAOAProblemMapping):
            raise TypeError("Problem must be an instance of QAOAProblemMapping")
        if params is not None and not isinstance(params, ndarray):
            raise TypeError("Parameters must be an ndarray")

        p = self.qaoa_config.p
        shots = self.qaoa_config.shots
        device_name = self.qaoa_config.device_name

        # Build cost hamiltonian and mixer hamiltonian
        cost_h, mixer_h = problem.cost_and_mixer_hamiltonians()

        # Build QAOA circuit
        qaoa_circuit = QAOABuilder(p=p, cost_h=cost_h, mixer_h=mixer_h)
        
        # Set up device
        graph = problem.get_graph()
        wires = list(graph.nodes())
        dev = device(name=device_name, wires=wires, shots=shots)

        # Build cost QNode
        cost_qnode = qaoa_circuit.build_qnode(dev, return_counts=True)

        # Evaluate counts with oprimized parameters
        counts = cost_qnode(params=params)

        return counts

class CorrelationPreparation:
    """Class to prepare correlation matrices using QAOA states.
    Attributes:
        qaoa_config (QAOAConfig): Configuration for QAOA.
    Methods:
        prepare_correlation_matrix(graph: Graph, params: ndarray = None) -> ndarray:
            Prepare the correlation matrix using QAOA states.
        build_correlation_matrix(results: ndarray, G: Graph) -> ndarray:
            Build the correlation matrix from QAOA measurement results.
    """
    def __init__(self, qaoa_config: QAOAConfig):
        """Constructor for CorrelationPreparation.
        Args:
            qaoa_config (QAOAConfig): Configuration for QAOA.
        """
        self.qaoa_config = qaoa_config

    def prepare_correlation_matrix(self, problem: Problem, params: ndarray = None) -> list[tuple[float, tuple[int, int]]]:

        if not isinstance(problem, QAOAProblemMapping):
            raise TypeError("Problem must be an instance of QAOAProblemMapping")
        if params is not None and not isinstance(params, ndarray):
            raise TypeError("Parameters must be an ndarray")

        device_name = self.qaoa_config.device_name
        p = self.qaoa_config.p
        shots = self.qaoa_config.shots

        # Build cost hamiltonian and mixer hamiltonian
        cost_h, mixer_h = problem.cost_and_mixer_hamiltonians()

        # Build QAOA circuit
        qaoa_circuit = QAOABuilder(p=p, cost_h=cost_h, mixer_h=mixer_h)
        
        # Set up device
        graph = problem.get_graph()
        wires = list(graph.nodes())
        dev = device(name=device_name, wires=wires, shots=shots)

        # Build correlation QNode
        correlation_qnode = qaoa_circuit.build_correlations_qnode(dev, graph)

        # Evaluate correlations with oprimized parameters
        correlations = correlation_qnode(params=params)

        # Return correlation matrix
        return CorrelationPreparation._build_corr_entries(correlations, graph)
    
    @staticmethod
    def _build_corr_entries(results: ndarray, G: Graph) -> list[tuple[float, tuple[int, int]]]:
        """Build the sorted correlation entries from QAOA measurement results.
        Note: Better to return a list of tuples instead of a full matrix for easier processing.
        Args:
            results (ndarray): The QAOA measurement results.
            G (Graph): The input graph.
        Returns:
            list[tuple[float, tuple[int, int]]]: The correlation entries as a list of tuples.
        """
        node_list = sorted(G.nodes())
        edge_list = sorted((min(u, v), max(u, v)) for (u, v) in G.edges())

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
