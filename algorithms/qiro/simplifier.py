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
    
    def simplify(self, problem: Problem, updated_solution: set[int], correlation_dict: list[tuple[float, tuple[int, int]]]):
        """
        Apply simplification rules to the given problem and update the solution set.

        Args:
            problem (Problem): The input problem for the combinatorial optimization problem.
            updated_solution (set[int]): The current solution set to be updated.
            correlation_dict (list[tuple[float, tuple[int, int]]]): The correlation values with corresponding node pairs.
        
        Raises:
            NoApplicableSimplificationRuleError: If no applicable simplification rule is found.
        """
        # Iterate over correlation entries and try to apply the simplification rules.
        # If rule is applied, return immediately. If the rule is not applicable, continue to the next entry.
        # It raises an error if no rule is applicable.
        for correlation, (u, v) in correlation_dict:

            if u == v:
                rule_applied = self.one_point_rule.apply(u, correlation, problem, updated_solution)
            else:
                rule_applied = self.two_points_rule.apply(u, v, correlation, problem, updated_solution)
            
            if rule_applied:
                return
        
        raise NoApplicableSimplificationRuleError("No applicable simplification rule found.")

class NoApplicableSimplificationRuleError(RuntimeError):
    """Exception raised when no applicable simplification rule is found."""
    def __init__(self, message: str):
        super().__init__(message)
