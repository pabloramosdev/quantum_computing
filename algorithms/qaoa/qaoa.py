from ..commons.state_preparation import StatePreparation
from ..problems import Problem

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

    def solve(self, problem: Problem) -> str:

        if not isinstance(problem, Problem):
            raise TypeError("El problema debe ser una instancia de Problem")

        optimized_params = self.state_preparation.prepare_qaoa_state(problem)
        
        counts = self.state_preparation.get_counts(problem=problem, params=optimized_params)
        
        max_bitstring = max(counts, key=counts.get)

        return {i for i, bit in enumerate(max_bitstring) if bit == "1"}
