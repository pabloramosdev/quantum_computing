from .reducer import Reducer
from ..problems import Problem

class QIROSolver:
    """Class to solve combinatorial optimization problems using QIRO.
    Attributes:
        reducer (Reducer): The reducer instance to apply reduction rules.
    Methods:
        solve(problem: Problem) -> set[int]:
            Apply the reduction process to the given problem and return the solution set.
    """
    
    def __init__(self, reducer: Reducer):
        """Constructor for QIROSolver.
        Args:
            reducer (Reducer): The reducer instance to apply reduction rules.
        """
        self.reducer = reducer
    
    def solve(self, problem: Problem) -> set[int]:
        """
        Apply the reduction process to the given graph and return the solution set.
        Args:
            problem (Problem): The input problem for the combinatorial optimization problem.
        Returns:
            set[int]: The solution as a set of node indices.
        """
        updated_problem = problem.copy()
        solution = set()

        num_iteration=0
        while updated_problem.number_of_edges() > 0:
            # Apply reduction rules
            print(f"Iteration {num_iteration}\n")
            self.reducer.reduce(updated_problem, solution)
            num_iteration += 1
        return solution
