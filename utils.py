import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from networkx import (
    Graph, 
    erdos_renyi_graph,
    draw_kamada_kawai, 
    is_connected
)


def create_connected_erdos_renyi_graph(n: int, p: float, seed: int = 42) -> tuple[Graph, int]:
    """
    Create a connected Erdős-Rényi graph with n nodes and edge probability p.
    If the generated graph is not connected, it will retry with different seeds
    until a connected graph is found.

    Args:
        n: Number of nodes in the graph.
        p: Probability of edge creation between nodes.
        seed: Initial seed for random number generation.

    Returns:
        A tuple containing the connected Erdős-Rényi graph and the seed used.
    """
    trial = 0
    while True:
        seed_effective = seed + trial
        G = erdos_renyi_graph(n, p, seed=seed_effective)
        if is_connected(G):
            return G, seed_effective
        trial += 1
    
def is_vertex_cover(G: Graph, cover: set[int]) -> bool:
    """
    Check if the given set 'cover' is a vertex cover of graph G.
    A vertex cover is a set of vertices such that every edge in the graph
    has at least one node in the set.
    
    Args:
        G: A NetworkX graph.
        cover: A set of vertices.
    
    Returns:
        True if 'cover' is a vertex cover of G, False otherwise.
    """
    return all(u in cover or v in cover for u, v in G.edges())

def uncovered_edges(G: Graph, cover: set[int]) -> list[tuple[int, int]]:
    """
    Return a list of edges that are not covered by the given vertex cover.
    Args:
        G: A NetworkX graph.
        cover: A set of vertices.
    
    Returns:
        A list of edges (tuples) that are not covered by 'cover'.    
    """
    return [(u, v) for u, v in G.edges() if u not in cover and v not in cover]


def is_independent_set(G: Graph, indep: set[int]) -> bool:
    return all(not (u in indep and v in indep) for u, v in G.edges())

def violating_edges(G: Graph, indep: set[int]) -> list[tuple[int, int]]:
    return [(u, v) for u, v in G.edges() if u in indep and v in indep]


def show_solution(G: Graph, cover: set[int], title: str, text_orange: str = "Inside Solution", text_gray: str = "Outside solution", legend: bool = True) -> None:
    """
    Visualize the graph G highlighting the vertices in the vertex cover.
        Args:
            G: A NetworkX graph.
            cover: A set of vertices representing the vertex cover.
            title: Title for the plot.
            legend: Whether to display the legend.
    """
    node_colors = ['orange' if n in cover else 'lightgray' for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(5, 4))

    draw_kamada_kawai(
        G,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
    )

    ax.set_title(title)

    if legend:
        orange_patch = mpatches.Patch(color="orange", label=text_orange)
        gray_patch = mpatches.Patch(color="lightgray", label=text_gray)
        ax.legend(handles=[orange_patch, gray_patch], loc="lower left")

    fig.tight_layout()
    plt.show()