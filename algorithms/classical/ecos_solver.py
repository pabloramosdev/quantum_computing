# Library cvxpy allows us to formulate optimization problems mathematically
# It works with various solvers; in this module, we will use the ECOS_BB solver
from networkx import Graph

from cvxpy import (
    Variable, 
    Problem, 
    Maximize, 
    Minimize, 
    sum, 
    ECOS_BB
)

def solve_vertex_cover(G: Graph) -> set[int]:
    """Solve the Minimum Vertex Cover.

    Model:
        minimize    sum_i x_i
        s.t.        x_u + x_v >= 1   for all edges (u,v)
                    x_i ∈ {0,1}

    Assumption:
        Nodes labeled in 0..n-1 

    Returns:
        set[int]: indices of nodes in the vertex cover.
    """

    n = G.number_of_nodes()
    x = Variable(n, boolean=True)

    constraints = [x[u] + x[v] >= 1 for u, v in G.edges()]
    prob = Problem(Minimize(sum(x)), constraints)
    prob.solve(solver=ECOS_BB)
    
    return {i for i in range(n) if x.value[i] > 0.5}


def solve_max_independent_set(G: Graph) -> set[int]:
    """Solve the Maximum Independent Set.

    Model:
        maximize    sum_i x_i
        s.t.        x_u + x_v <= 1   for all edges (u,v)
                    x_i ∈ {0,1}

    Assumption:
        Nodes labeled in 0..n-1 

    Returns:
        set[int]: indices of nodes in the independent set.
    """
    n = G.number_of_nodes()
    x = Variable(n, boolean=True)

    constraints = [x[u] + x[v] <= 1 for u, v in G.edges()]
    prob = Problem(Maximize(sum(x)), constraints)
    prob.solve(solver=ECOS_BB)

    return {i for i in range(n) if x.value[i] > 0.5}
