# Quantum and Classical Approaches for the Minimum Vertex Cover Problem

This repository provides a comparative implementation of quantum and classical algorithms
for solving the **Minimum Vertex Cover (MVC)** problem on graphs.

The project includes:

- A **quantum implementation of QAOA** (Quantum Approximate Optimization Algorithm),
- A **quantum-informed recursive algorithm (QIRO)** built on top of QAOA correlations,
- A **classical exact solver** based on convex optimization using **CVXPY with the ECOS_BB solver**.

All quantum algorithms are implemented using **PennyLane**, enabling execution on simulators
and, in principle, on quantum hardware.

---

## 📌 Algorithms Implemented

### 🔹 QAOA (Quantum Approximate Optimization Algorithm)

QAOA is a variational quantum algorithm designed to approximately solve combinatorial
optimization problems by alternating between:

- a **cost Hamiltonian** encoding the MVC objective, and
- a **mixer Hamiltonian** that explores the solution space.

The implementation provided in this repository is modular and configurable, allowing
different circuit depths, Hamiltonians, and optimization strategies.

---

### 🔹 QIRO (Quantum-Informed Recursive Optimization)

QIRO is a hybrid quantum–classical algorithm that extends QAOA by exploiting information
extracted from shallow quantum circuits, in particular **one-point and two-point correlation
functions**, to iteratively reduce the size of a combinatorial optimization problem.

The original QIRO algorithm was introduced in:

J. Finžgar et al., *Quantum-Informed Recursive Optimization Algorithms*,  
PRX Quantum 5, 020327 (2024).  
https://doi.org/10.1103/PRXQuantum.5.020327

The implementation provided in this repository is **inspired by the conceptual ideas described
in the original QIRO paper**, but it is **not a copy** of the original algorithm nor of any
existing reference implementation.

In particular:

- The implementation has been **developed independently from scratch**.
- No source code from the original authors or from any third-party repository has been reused.
- The software architecture, data structures, and simplification routines are **original**.
- While the algorithm follows the *principles* described in the paper, the concrete realization
  differs in structure and implementation details.

As such, the QIRO implementation in this repository constitutes an **original contribution**
intended for experimentation, analysis, and further research.

---

### 🔹 Classical Solver (CVXPY + ECOS_BB)

As a classical baseline, the Minimum Vertex Cover problem is also solved **exactly** using:

- **CVXPY** for problem formulation,
- the **ECOS_BB** mixed-integer solver.

This solver is used to obtain reference solutions for comparison with the quantum approaches.

---

## 🧪 Example Usage

The repository includes a Jupyter notebook:

Minimum Vertex Cover.ipynb

This notebook demonstrates:
- how to define a graph instance,
- how to run **QAOA**, **QIRO**, and the **classical solver**,
- and how to compare their results.

It serves as a **self-contained example** of how to use the three approaches implemented
in this repository.

---

## 🐍 Python Version

The code in this repository has been developed and tested using:

- **Python 3.12.10**

Compatibility with other Python versions has not been tested.

---

## 📦 Requirements

All required Python packages and their tested versions are listed in `requirements.txt`.

The main dependencies include:
- NumPy
- SciPy
- Matplotlib
- PennyLane
- CVXPY
- ECOS

---

## 🛠️ Installation Guide

### 1️⃣ Check Python version

Make sure you are using the correct Python version:

python --version

---

### 2️⃣ (Recommended) Create a virtual environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

Linux / macOS:
source venv/bin/activate

---

### 3️⃣ Install dependencies

Upgrade pip and install the required packages:

pip install --upgrade pip  
pip install -r requirements.txt

---

### 4️⃣ Verify the installation

You can verify that the main dependencies are correctly installed by running:

python -c "import pennylane, cvxpy, networkx, numpy, scipy, matplotlib; print('Environment OK')"

---

### 5️⃣ Run the example notebook

Launch Jupyter and open the example notebook:

jupyter notebook

Then run:

Minimum Vertex Cover.ipynb

---

## 🎯 Purpose of the Project

This repository is intended for:
- research and experimentation with variational quantum algorithms,
- studying hybrid quantum–classical optimization strategies,
- benchmarking quantum algoritms against exact classical solvers.

The code is written with clarity and modularity in mind, making it suitable as a starting
point for further research or extensions.

---

## ✉️ Contact

If you are interested in this work or would like to discuss extensions or collaborations,
feel free to get in touch.
