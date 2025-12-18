from networkx import Graph

from ..commons.state_preparation import StatePreparation

class QAOASolver:
    """Class to solve combinatorial optimization problems using QAOA.
    Attributes:
        state_preparation (StatePreparation): The state preparation instance for QAOA.
    Methods:
        solve(graph: Graph) -> str:
            Solve the combinatorial optimization problem on the given graph.
    """
    def __init__(self, state_preparation: StatePreparation):
        """Constructor for QAOASolver.
        Args:
            state_preparation (StatePreparation): The state preparation instance for QAOA.
        """
        self.state_preparation = state_preparation

    def solve(self, graph: Graph) -> str:
        """Solve the combinatorial optimization problem on the given graph.
        Args:
            graph (Graph): The input graph for the combinatorial optimization problem.
        Returns:
            str: The solution as a set of node indices.
        Raises:
            TypeError: If the input graph is not a networkx.Graph instance.
        """
        if not isinstance(graph, Graph):
            raise TypeError("El grafo debe ser una instancia de networkx.Graph")

        optimized_params = self.state_preparation.prepare_qaoa_state(graph)
        
        counts = self.state_preparation.get_counts(graph=graph, params=optimized_params)
        
        max_bitstring = max(counts, key=counts.get)

        return {i for i, bit in enumerate(max_bitstring) if bit == "1"}
