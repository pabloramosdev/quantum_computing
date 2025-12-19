from networkx import Graph

from .simplifier import Simplifier

from ..commons.state_preparation import StatePreparation, CorrelationPreparation

class Reducer:
    """ Reducer class for combinatorial optimization problems.
    Attributes:
        simplifier (Simplifier): The simplifier instance to apply simplification rules.
        state_preparation (StatePreparation): The state preparation instance for QAOA.
        correlation_preparation (CorrelationPreparation): The correlation preparation instance for QAOA.
    Methods:
        reduce(simplified_problem: Graph, updated_solution: set[int]) -> tuple[Graph, set[int]]:
            Reduce routine for combinatorial optimization problems.
    """

    def __init__(self, simplifier: Simplifier, state_preparation: StatePreparation, correlation_preparation: CorrelationPreparation):
        """Constructor for Reducer.
        Args:
            simplifier (Simplifier): The simplifier instance to reduce the graph.
            state_preparation (StatePreparation): The state preparation instance for QAOA.
            correlation_preparation (CorrelationPreparation): The correlation preparation instance for QAOA.
        """
        self.simplifier = simplifier
        self.state_preparation = state_preparation
        self.correlation_preparation = correlation_preparation

    def reduce(self, simplified_problem: Graph, updated_solution: set[int]) -> tuple[Graph,  set[int]]:
        """        
        Reduce routine for combinatorial optimization problems.
        Args:
            simplified_problem (Graph): The simplified problem graph.
            updated_solution (set[int]): The current solution set of node indices.
        Returns:
            tuple[Graph, set[int]]: The reduced problem graph and updated solution set.
        """
        # Prepare the QAOA state and optimize parameters
        optimized_params = self.state_preparation.prepare_qaoa_state(graph=simplified_problem)

        # Build the correlation matrix with the optimal parameters
        correlation_matrix = self.correlation_preparation.prepare_correlation_matrix(graph=simplified_problem, params=optimized_params)

        # Apply simplification rules to reduce the graph and update the solution
        simplified_problem, updated_solution = self.simplifier.simplify(simplified_problem, updated_solution, correlation_matrix)
        
        return simplified_problem, updated_solution
