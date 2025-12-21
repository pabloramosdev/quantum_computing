from abc import ABC, abstractmethod

from ..problems import Problem

class OnePointRule(ABC):
    """
    OnePointRule abstract class for combinatorial optimization problem.
    Applies an action based on the correlation of a node.
    This is an abstract class and must be implemented by subclasses.
    """

    @abstractmethod
    def apply(self, node: int, correlation: float, problem: Problem, updated_solution: set[int]):
        pass

class TwoPointsRule(ABC):
    """
    TwoPointsRule abstract class for combinatorial optimization problem.
    Applies an action based on the correlation of an edge.
    This is an abstract class and must be implemented by subclasses.
    """

    @abstractmethod
    def apply(self, u: int, v: int, correlation: float, problem: Problem, updated_solution: set[int]):
        pass


class VertexCoverOnePointRule(OnePointRule):
    """ Rule of one point for the Vertex Cover problem.
    """
    def apply(self, node: int, correlation: float, problem: Problem, updated_solution: set[int]):
        """
        Apply the one-point rule for Vertex Cover problem.
        Args:
            node (int): The node index.
            correlation (float): The correlation value ⟨Z⟩ for the node.
            problem (Problem): The input problem for the combinatorial optimization problem.
            updated_solution (set[int]): The current solution set to be updated.
        """
        if correlation >= 0:
            neighbors = list(problem.neighbors(node))
            updated_solution.update(neighbors)
            problem.remove_nodes(neighbors + [node])
            print(f"[MVC-3] Node {node} with ⟨Z⟩ = {correlation:.4f} -> neighbors {neighbors} added to cover, node {node} and neighbors removed from graph.")
        else:
            updated_solution.add(node)
            problem.remove_node(node)
            print(f"[MVC-4] Node {node} with ⟨Z⟩ = {correlation:.4f} -> added to cover.")
        
        isolated = [n for n in problem.nodes() if problem.degree(n) == 0]
        problem.remove_nodes(isolated)

        if isolated:
            print(f"Isolated nodes removed: {isolated}")

class VertexCoverTwoPointsRule(TwoPointsRule):
    """Rule of two points for the Vertex Cover problem.
    """
    def apply(self, u: int, v: int, correlation: float, problem: Problem, updated_solution: set[int]):
        """
        Apply the two-points rule for Vertex Cover problem.
        Args:
            u (int): The first node index.
            v (int): The second node index.
            correlation (float): The correlation value ⟨ZZ⟩ for the edge (u, v).
            problem (Problem): The input problem for the combinatorial optimization problem.
            updated_solution (set[int]): The current solution set to be updated.
        """

        if correlation < 0:
            chosen = u if problem.degree(u) >= problem.degree(v) else v
            updated_solution.add(chosen)
            problem.remove_node(chosen)
            print(f"[MVC-1] Edge ({u}, {v}) with correlation <ZZ> = {correlation:.4f} -> node {chosen} added to cover.")
        else:
            updated_solution.update([u, v])
            problem.remove_nodes([u, v])
            print(f"[MVC-2] Edge ({u}, {v}) with correlation <ZZ> = {correlation:.4f} -> nodes {u}, {v} added to cover.")
        
        isolated = [n for n in problem.nodes() if problem.degree(n) == 0]
        problem.remove_nodes(isolated)

        if isolated:
            print(f"Isolated nodes removed: {isolated}")
