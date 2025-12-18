from networkx import Graph

from numpy import (
    ndarray, 
    abs, 
    argmax, 
    unravel_index
)

from .rules import OnePointRule, TwoPointsRule

class Simplifier():
    """ Class to apply simplification rules for combinatorial optimization problems using correlation matrices.
    Attributes: 
        one_point_rule (OnePointRule): Rule to apply for one-point simplification.
        two_points_rule (TwoPointsRule): Rule to apply for two-points simplification.
    Methods:
        simplify(graph: Graph, solution_set: set[int], correlation_matrix: ndarray) -> tuple[Graph, set[int]]:
            Apply simplification rules to the graph based on the correlation matrix and update the solution set.
    """
    def __init__(self, one_point_rule: OnePointRule, two_points_rule: TwoPointsRule):
        """
        Constructor for Simplifier.
        Args:
            one_point_rule (OnePointRule): Rule to apply for one-point simplification.
            two_points_rule (TwoPointsRule): Rule to apply for two-points simplification.
        """
        self.one_point_rule = one_point_rule
        self.two_points_rule = two_points_rule

    def simplify(self, graph: Graph, solution_set: set[int], correlation_matrix: ndarray) -> tuple[Graph, set[int]]:
        """
        Apply simplification rules to the graph based on the correlation matrix and update the solution set.
        Args:
            graph (Graph): The input graph for the combinatorial optimization problem.
            solution_set (set[int]): The current solution set of node indices.
            correlation_matrix (ndarray): The correlation matrix obtained from optimal QAOA states.
        Returns:
            tuple[Graph, set[int]]: The reduced graph and updated solution set.
        """
        reduced_graph = graph.copy()
        updated_solution = set(solution_set)

        # Finds the index of the coordinate with the maximum absolute correlation value in the flattened matrix
        max_corr_flatten_index = argmax(abs(correlation_matrix))

        # Get the coordinates of the maximum correlation in the matrix
        u, v = unravel_index(max_corr_flatten_index, correlation_matrix.shape)

        # Get the correlation value at the coordinates
        correlation = correlation_matrix[u, v]

        if u == v:
            return self.one_point_rule.apply(int(u), correlation, reduced_graph, updated_solution)
        else:
            return self.two_points_rule.apply(int(u), int(v), correlation, reduced_graph, updated_solution)

