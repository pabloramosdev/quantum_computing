from networkx import Graph

from abc import ABC, abstractmethod

class OnePointRule(ABC):
    """
    OnePointRule abstract class for combinatorial optimization problem.
    Applies an action based on the correlation of a node.
    This is an abstract class and must be implemented by subclasses.
    """

    @abstractmethod
    def apply(self, node: int, correlation: float, reduced_graph: Graph, updated_solution: set[int]):
        """
        Applies the one-point rule to the given node.
        Args:
            node: Node of the graph.
            correlation: Correlation of the node with the solution set.
            reduced_graph: Reduced graph where the rules are applied.
            updated_solution: Updated solution set.
        Returns:
            reduced_graph: Updated reduced graph.
            updated_solution: Updated solution set.
        """
        pass

class TwoPointsRule(ABC):
    """
    TwoPointsRule abstract class for combinatorial optimization problem.
    Applies an action based on the correlation of an edge.
    This is an abstract class and must be implemented by subclasses.
    """

    @abstractmethod
    def apply(self, u: int, v: int, correlation: float, reduced_graph: Graph, updated_solution: set[int]):
        """
        Applies the two-points rule to the given edge.
        Args:
            u: First node of the edge.
            v: Second node of the edge.
            correlation: Correlation of the edge with the solution set.
            reduced_graph: Reduced graph where the rules are applied.
            updated_solution: Updated solution set.
        Returns:
            reduced_graph: Updated reduced graph.
            updated_solution: Updated solution set.
        """
        pass


class VertexCoverOnePointRule(OnePointRule):
    """ Rule of one point for the Vertex Cover problem.
    """
    def apply(self, node: int, correlation: float, reduced_graph: Graph, updated_solution: set[int]):
        """
        Applies the one-point rule to the given node.
        If the correlation is positive, the neighbors of the node are added to the solution set.
        If the correlation is negative, the node is added to the solution set and removed from the reduced graph.

        Args:
            node: Node of the graph.
            correlation: Correlation of the node with the solution set.
            reduced_graph: Reduced graph where the rules are applied.
            updated_solution: Updated solution set.
        Returns:
            reduced_graph: Updated reduced graph.
            updated_solution: Updated solution set.
        """
        if correlation >= 0:
            neighbors = list(reduced_graph.neighbors(node))
            updated_solution.update(neighbors)
            reduced_graph.remove_nodes_from(neighbors + [node])
            print(f"[MVC-3] Node {node} with ⟨Z⟩ = {correlation:.4f} -> neighbors {neighbors} added to cover, node {node} and neighbors removed from graph.")
        else:
            updated_solution.add(node)
            reduced_graph.remove_node(node)
            print(f"[MVC-4] Node {node} with ⟨Z⟩ = {correlation:.4f} -> added to cover.")
        
        isolated = [n for n in reduced_graph.nodes if reduced_graph.degree[n] == 0]
        reduced_graph.remove_nodes_from(isolated)

        if isolated:
            print(f"Isolated nodes removed: {isolated}")

        return reduced_graph, updated_solution

class VertexCoverTwoPointsRule(TwoPointsRule):
    """Rule of two points for the Vertex Cover problem.
    """
    def apply(self, u: int, v: int, correlation: float, reduced_graph: Graph, updated_solution: set[int]):
        """
        Applies the two-points rule to the given edge.
        If the correlation is negative, the node with higher degree is added to the solution set.
        If the correlation is positive, both nodes are added to the solution set.
        Args:
            u: First node of the edge.
            v: Second node of the edge.
            correlation: Correlation of the edge with the solution set.
            reduced_graph: Reduced graph where the rules are applied.
            updated_solution: Updated solution set.
        Returns:
            reduced_graph: Updated reduced graph.
            updated_solution: Updated solution set.
        """
        if correlation < 0:
            chosen = u if reduced_graph.degree[u] >= reduced_graph.degree[v] else v
            updated_solution.add(chosen)
            reduced_graph.remove_node(chosen)
            print(f"[MVC-1] Edge ({u}, {v}) with correlation <ZZ> = {correlation:.4f} -> node {chosen} added to cover.")
        else:
            updated_solution.update([u, v])
            reduced_graph.remove_nodes_from([u, v])
            print(f"[MVC-2] Edge ({u}, {v}) with correlation <ZZ> = {correlation:.4f} -> nodes {u}, {v} added to cover.")
        
        isolated = [n for n in reduced_graph.nodes if reduced_graph.degree[n] == 0]
        reduced_graph.remove_nodes_from(isolated)

        if isolated:
            print(f"Isolated nodes removed: {isolated}")
        
        return reduced_graph, updated_solution

