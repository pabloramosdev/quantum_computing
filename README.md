# Quantum Notebooks

This repository provides a collection of notebooks and reference implementations of quantum algorithms, with a particular focus on combinatorial optimization problems.

The main emphasis is placed on the Quantum Approximate Optimization Algorithm (QAOA) and the Quantum-Informed Recursive Optimization (QIRO) algorithm, along with supporting utilities for graph generation and visualization.

## Disclaimer

This repository is intended for educational and research purposes only.

The QIRO implementation provided in this repository is an independent and original implementation developed by the author. It is not affiliated with, endorsed by, or derived from any official or reference implementation of QIRO, nor from any other existing software implementation.

The implementation is based solely on the algorithms and methodological descriptions presented in the original QIRO research paper:

https://arxiv.org/abs/2308.13607

Any design choices, simplifications, extensions, or implementation details are the author’s own and may differ from those used in other implementations.

Furthermore, the provided implementations represent simplified versions of complex quantum algorithms and are not necessarily optimized for performance or scalability. Users are encouraged to further analyze, adapt, and refine these algorithms for practical or experimental applications.

The algorithms have been tested primarily on the Minimum Vertex Cover problem using graphs with up to 15 nodes. Most experiments were conducted using the PennyLane simulators `default.qubit` and `lightning.qubit`. Additionally, limited tests were performed using `lightning.kokkos` on Linux for graphs with up to 27 nodes.


## The structure of this repository is the following one:
- algorithms/: contains the implementation of quantum algorithms such as QAOA and QIRO for solving combinatorial optimization problems.
- utils.py: contains utility functions for graph generation, visualization, and evaluation of vertex covers.
- Minimum Vertex Cover.ipynb: A Jupyter notebook demonstrating the use of the implemented algorithms to solve the Minimum Vertex Cover problem on random graphs.
- max_cut_problem.ipynb: A Jupyter notebook demonstrating the use of the implemented algorithms to solve the Max-Cut problem on random graphs.
- quantum_support_vector_machine.ipynb: A Jupyter notebook demonstrating the use of quantum support vector machines.

## Minimum Vertex Cover Problem
The Minimum Vertex Cover (MVC) problem is a classic combinatorial optimization problem.
Given a graph $G = (V, E)$, the goal is to find the smallest subset of vertices $C \subseteq V$ such that every edge in $E$ is incident to at least one vertex in $C$.
This repository provides implementations of quantum algorithms such as QAOA and QIRO to tackle the MVC problem, along with classical method for comparison.

## Minimum Vertex Cover Notebook
The `Minimum Vertex Cover.ipynb` notebook demonstrates how to use the implemented quantum algorithms to solve the MVC problem on random graphs. It includes the following sections:
1. Graph Generation: Random graph is generated for testing the algorithms.
2. Classical Solution: A classical solver is used to find the optimal vertex cover for comparison. ECOS_BB is used as the classical solver.
3. QAOA Solution: The Quantum Approximate Optimization Algorithm (QAOA) is implemented to find approximate solutions to the MVC problem.
4. QIRO Solution: The Quantum-Informed Recursive Optimization (QIRO) algorithm is implemented to improve the solutions obtained from QAOA.
5. Visualization: The results are visualized, showing the original graph and the vertex covers found by the algorithms.

## Requirements
To run the notebooks and use the implementations, you need to have Python installed:
- Python 3.10 or higher

## Installation
1. Clone the repository:
   ```bash
    git clone https://github.com/pabloramosdev/quantum_computing.git
    cd quantum_computing
    ```
2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
## Usage
To run the Minimum Vertex Cover notebook, navigate to the repository directory and launch Jupyter Notebook:

```bash
jupyter notebook "Minimum Vertex Cover.ipynb"
```