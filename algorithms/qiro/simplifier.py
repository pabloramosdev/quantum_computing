from numpy import (
    ndarray, 
    abs, 
    argmax, 
    unravel_index
)

from .rules import OnePointRule, TwoPointsRule

from ..problems import Problem

class Simplifier():
    """Class to simplify combinatorial optimization problems using reduction rules.
    Attributes:
        one_point_rule (OnePointRule): Rule to apply for one-point simplification.
        two_points_rule (TwoPointsRule): Rule to apply for two-points simplification.
    Methods:
        simplify(problem: Problem, updated_solution: set[int], correlation_matrix: ndarray):
            Apply simplification rules to the given problem and update the solution set.
    """
    def __init__(self, one_point_rule: OnePointRule, two_points_rule: TwoPointsRule):
        """Constructor for Simplifier.
        Args:
            one_point_rule (OnePointRule): Rule to apply for one-point simplification.
            two_points_rule (TwoPointsRule): Rule to apply for two-points simplification.
        """
        self.one_point_rule = one_point_rule
        self.two_points_rule = two_points_rule

    def simplify(self, problem: Problem, updated_solution: set[int], correlation_matrix: ndarray):
        """
        Apply simplification rules to the given problem and update the solution set.
        Args:
            problem (Problem): The input problem for the combinatorial optimization problem.
            updated_solution (set[int]): The current solution set to be updated.
            correlation_matrix (ndarray): The correlation matrix used for simplification.
        """
        # Finds the index of the coordinate with the maximum absolute correlation value in the flattened matrix
        max_corr_flatten_index = argmax(abs(correlation_matrix))

        # Get the coordinates of the maximum correlation in the matrix
        u, v = unravel_index(max_corr_flatten_index, correlation_matrix.shape)

        # Get the correlation value at the coordinates
        correlation = correlation_matrix[u, v]

        if u == v:
            return self.one_point_rule.apply(int(u), correlation, problem, updated_solution)
        else:
            return self.two_points_rule.apply(int(u), int(v), correlation, problem, updated_solution)
