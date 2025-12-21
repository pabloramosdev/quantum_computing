from abc import ABC, abstractmethod

from networkx import Graph

from pennylane import Hamiltonian
from pennylane.qaoa.cost import (
    maxcut, 
    max_clique,
    min_vertex_cover,
    max_weight_cycle,
    max_independent_set
)

class Problem:

    def __init__(self, graph: Graph):
        self.graph = graph

    def remove_node(self, node: int) -> None:
        self.graph.remove_node(node)
    
    def remove_nodes(self, nodes: set[int]) -> None:
        self.graph.remove_nodes_from(nodes)
    
    def remove_edge(self, u: int, v: int) -> None:
        self.graph.remove_edge(u, v)
    
    def remove_edges(self, edges: set[tuple[int, int]]) -> None:
        self.graph.remove_edges_from(edges)
    
    def get_graph(self) -> Graph:
        return self.graph

    def degree(self, node: int) -> int:
        return self.graph.degree[node]

    def neighbors(self, node: int) -> list[int]:
        return list(self.graph.neighbors(node))

    def nodes(self) -> list[int]:
        return list(self.graph.nodes)
    
    def edges(self) -> list[tuple[int, int]]:
        return list(self.graph.edges)

    def number_of_nodes(self) -> int:
        return self.graph.number_of_nodes()
    
    def number_of_edges(self) -> int:
        return self.graph.number_of_edges()
    
    def copy(self):
        return (type(self))(self.graph.copy())

class QAOAProblemMapping(ABC, Problem):
    
    @abstractmethod
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        pass

class MaxCut(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return maxcut(self.graph)

class MaxCliqueXMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return max_clique(self.graph)

class MaxCliqueBitFlipMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return max_clique(self.graph, constrained=True)

class MinVertexCoverXMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return min_vertex_cover(self.graph)

class MinVertexCoverBitFlipMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return min_vertex_cover(self.graph, constrained=True)

class MaxWeightCycleXMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return max_weight_cycle(self.graph)

class MaxWeightCycleBitFlipMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return max_weight_cycle(self.graph, constrained=True)

class MaxIndependentSetXMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return max_independent_set(self.graph)

class MaxIndependentSetBitFlipMixer(QAOAProblemMapping):

    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return max_independent_set(self.graph, constrained=True)