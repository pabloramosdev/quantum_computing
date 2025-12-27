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
    """ Base class for graph-based optimization problems.

        Note: Wrapper around NetworkX Graph to facilitate graph manipulations.

        Attributes:
            graph (Graph): The graph representing the problem instance.
        
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def remove_node(self, node: int) -> None:
        """Removes a node from the graph.

            Args:
                node (int): The node to be removed.
        """
        self.graph.remove_node(node)
    
    def remove_nodes(self, nodes: set[int]) -> None:
        """Removes multiple nodes from the graph.

            Args:
                nodes (set[int]): The nodes to be removed.
        """
        self.graph.remove_nodes_from(nodes)
    
    def remove_edge(self, u: int, v: int) -> None:
        """Removes an edge from the graph.

            Args:
                u (int): One endpoint of the edge to be removed.
                v (int): The other endpoint of the edge to be removed.
        """
        self.graph.remove_edge(u, v)
    
    def remove_edges(self, edges: set[tuple[int, int]]) -> None:
        """Removes multiple edges from the graph.

            Args:
                edges (set[tuple[int, int]]): The edges to be removed.
        """
        self.graph.remove_edges_from(edges)

    def degree(self, node: int) -> int:
        """Returns the degree of a node in the graph.
            
            Args:
                node (int): The node whose degree is to be returned.

            Returns:
                int: The degree of the node.
        """
        return int(self.graph.degree(node))

    def neighbors(self, node: int) -> list[int]:
        """Returns the neighbors of a node in the graph.
            
            Args:
                node (int): The node whose neighbors are to be returned.

            Returns:
                list[int]: The neighbors of the node.
        """
        return list(self.graph.neighbors(node))

    def nodes(self) -> list[int]:
        """Returns the nodes of the graph.
            
            Returns:
                list[int]: The nodes of the graph.
        """
        return list(self.graph.nodes())
    
    def edges(self) -> list[tuple[int, int]]:
        """Returns the edges of the graph.
            
            Returns:
                list[tuple[int, int]]: The edges of the graph.
        """
        return list(self.graph.edges())

    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the graph.
            
            Returns:
                int: The number of nodes in the graph.
        """
        return self.graph.number_of_nodes()
    
    def number_of_edges(self) -> int:
        """Returns the number of edges in the graph.
            
            Returns:
                int: The number of edges in the graph.
        """
        return self.graph.number_of_edges()
    
    def copy(self):
        """Creates a copy of the problem instance.
            
            Returns:
                Problem: A copy of the problem instance.
        """
        return (type(self))(self.graph.copy())

class QAOAProblemMapping(ABC, Problem):
    
    @abstractmethod
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        pass

class MaxCut(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        """Returns the cost and mixer Hamiltonians for the Max-Cut problem on the graph.

        Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.maxcut.html
        
        Returns:
            tuple[Hamiltonian, Hamiltonian]: The cost and mixer Hamiltonians.
        """
        return maxcut(self.graph)

class MaxCliqueXMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        """Returns the cost and mixer Hamiltonians for the Max-Clique problem on the graph.
        
        Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.max_clique.html

        Mixer: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.mixers.x_mixer.html
        
        Returns:
            tuple[Hamiltonian, Hamiltonian]: The cost and mixer Hamiltonians.
        """
        return max_clique(self.graph)

class MaxCliqueBitFlipMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        """Returns the cost and mixer Hamiltonians for the Max-Clique problem on the graph.
        
        Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.max_clique.html

        Mixer: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.mixers.bit_flip_mixer.html

        Returns:
            tuple[Hamiltonian, Hamiltonian]: The cost and mixer Hamiltonians.
        """
        return max_clique(self.graph, constrained=True)

class MinVertexCoverXMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        """Returns the cost and mixer Hamiltonians for the Min-Vertex-Cover problem on the graph.
        
        Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.min_vertex_cover.html
        
        Mixer: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.mixers.x_mixer.html

        Returns:
            tuple[Hamiltonian, Hamiltonian]: The cost and mixer Hamiltonians.
        """

        return min_vertex_cover(self.graph)

class MinVertexCoverBitFlipMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        """Returns the cost and mixer Hamiltonians for the Min-Vertex-Cover problem on the graph.
        
        Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.min_vertex_cover.html
        
        Mixer: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.mixers.bit_flip_mixer.html
        
        Returns:
            tuple[Hamiltonian, Hamiltonian]: The cost and mixer Hamiltonians.
        """
        return min_vertex_cover(self.graph, constrained=True)

class MaxWeightCycleXMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        """Returns the cost and mixer Hamiltonians for the Max-Weight-Cycle problem on the graph.
        
        Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.max_weight_cycle.html

        Mixer: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.mixers.x_mixer.html

        Returns:
            tuple[Hamiltonian, Hamiltonian]: The cost and mixer Hamiltonians.
        """
        return max_weight_cycle(self.graph)

class MaxWeightCycleConstrainedMixer(QAOAProblemMapping):
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        """Returns the cost and mixer Hamiltonians for the Max-Weight-Cycle problem on the graph.

        Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.max_weight_cycle.html
        
        Mixer: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cycle.cycle_mixer.html
        
        Returns:
            tuple[Hamiltonian, Hamiltonian]: The cost and mixer Hamiltonians.
        """
        return max_weight_cycle(self.graph, constrained=True)

class MaxIndependentSetXMixer(QAOAProblemMapping):
    """Returns the cost and mixer Hamiltonians for the Max-Independent-Set problem on the graph.

        Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.max_independent_set.html

        Mixer: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.mixers.x_mixer.html

        Returns:
            tuple[Hamiltonian, Hamiltonian]: The cost and mixer Hamiltonians.
    """
    
    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        return max_independent_set(self.graph)

class MaxIndependentSetBitFlipMixer(QAOAProblemMapping):

    def cost_and_mixer_hamiltonians(self) -> tuple[Hamiltonian, Hamiltonian]:
        """Returns the cost and mixer Hamiltonians for the Max-Independent-Set problem on the graph.
        
        Reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.max_independent_set.html
        
        Mixer: https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.mixers.bit_flip_mixer.html
        
        Returns:
            tuple[Hamiltonian, Hamiltonian]: The cost and mixer Hamiltonians.
        """
        return max_independent_set(self.graph, constrained=True)