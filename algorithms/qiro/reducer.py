from .simplifier import Simplifier
from ..commons.state_preparation import StatePreparation
from ..problems import Problem

class Reducer:
    """Class to reduce combinatorial optimization problems using QAOA-based simplification.
    Attributes:
        simplifier (Simplifier): The simplifier instance to reduce the graph.
        state_preparation (StatePreparation): The state preparation instance for QAOA.
    Methods:
        reduce(simplified_problem: Problem, updated_solution: set[int]):
            Apply reduction rules to the given problem and update the solution set.
    """
    def __init__(self, simplifier: Simplifier, state_preparation: StatePreparation):
        """Constructor for Reducer.
        Args:
            simplifier (Simplifier): The simplifier instance to reduce the graph.
            state_preparation (StatePreparation): The state preparation instance for QAOA.
        """
        self.simplifier = simplifier
        self.state_preparation = state_preparation
    def reduce(self, simplified_problem: Problem, updated_solution: set[int]):
        """
        Apply reduction rules to the given problem and update the solution set.
        Args:
            simplified_problem (Problem): The input problem for the combinatorial optimization problem.
            updated_solution (set[int]): The current solution set to be updated.
        """
        # Prepare the QAOA state and optimize parameters
        optimized_params = self.state_preparation.optimize(problem=simplified_problem)

        # Build the correlation matrix with the optimal parameters
        correlation_entries = self.state_preparation.correlations(problem=simplified_problem, params=optimized_params)

        # Apply simplification rules to reduce the graph and update the solution
        self.simplifier.simplify(simplified_problem, updated_solution, correlation_entries)
    