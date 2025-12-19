from networkx import Graph

from .reducer import Reducer

class QIROSolver:
    """ Class to solve combinatorial optimization problems using QIRO.
    Attributes:
        reducer (Reducer): The reducer instance to apply reduction rules.
    Methods:
        solve(graph: Graph) -> set[int]:
            Apply the reduction process to the given graph and return the solution set.
    """
    def __init__(self, reducer: Reducer):
        """ Constructor for QIROSolver.
        Args:
            reducer (Reducer): The reducer instance to apply reduction rules.
        """
        self.reducer = reducer
    
    def solve(self, graph: Graph) -> set[int]:
        """
        Apply the reduction process to the given graph and return the solution set.
        Args:
            graph (Graph): The input graph for the combinatorial optimization problem.
        Returns:
            set[int]: The solution as a set of node indices.
        """
        S = set()
        P_current = graph.copy()

        num_iteration=0
        while P_current.number_of_edges() > 0:
            # Apply reduction rules
            print(f"Iteration {num_iteration}\n")
            P_current, S = self.reducer.reduce(P_current, S)
            num_iteration += 1
        return S