# Library cvxpy allows us to formulate optimization problems mathematically
# It works with various solvers; in this notebook, we will use the ECOS_BB solver
import cvxpy as cp

def solve_vertex_cover(G):
    """ Solve the Minimum Vertex Cover problem using cvxpy.
    Args:
        G (Graph): The input graph for the Minimum Vertex Cover problem.
    Returns:
        set[int]: The solution as a set of node indices forming the vertex cover.
    """
    n = G.number_of_nodes()
    x = cp.Variable(n, boolean=True)
    constraints = [x[u] + x[v] >= 1 for u, v in G.edges()]
    prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
    prob.solve(solver=cp.ECOS_BB)
    return {i for i in range(n) if x.value[i] > 0.5}