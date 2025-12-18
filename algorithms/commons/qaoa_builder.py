from networkx import Graph
from numpy import ndarray

from pennylane import (
    Hamiltonian,
    Hadamard,
    PauliZ,
    counts,
    expval,
    layer,
    qnode
)

from pennylane.devices import Device
from pennylane.wires import Wires

from pennylane.math import concatenate, stack
from pennylane.qaoa import cost_layer, mixer_layer


class QAOABuilder:
    """Class to build QAOA circuits and QNodes.
    
    Attributes:
        p (int): Number of QAOA layers.
        cost_h (Hamiltonian): Cost Hamiltonian for the QAOA circuit.
        mixer_h (Hamiltonian): Mixer Hamiltonian for the QAOA circuit.
        
    Methods:
        build_qnode(device: Device, return_counts: bool = False):
            Build the QAOA qnode for expectation value or counts measurement.
        
        build_correlations_qnode(device: Device, G: Graph):
            Build the QAOA qnode for correlation measurements.
        _qaoa_layer(gamma: float, beta: float):
            Build a single layer of the QAOA circuit.
        _qaoa_circuit(params: ndarray, wires: Wires = None):
            Build the full QAOA circuit with p layers.
    
    """

    def __init__(self, p: int = 1, cost_h: Hamiltonian = None, mixer_h: Hamiltonian = None):
        """
        Class constructor for QAOABuilder.
        Args:
            p (int): Number of QAOA layers.
            cost_h (Hamiltonian): Cost Hamiltonian for the QAOA circuit.
            mixer_h (Hamiltonian): Mixer Hamiltonian for the QAOA circuit.
        Raises:
            ValueError: If p is not a positive integer or if hamiltonians are not provided.
            TypeError: If hamiltonians are not instances of Hamiltonian.
        """

        if p <= 0:
            raise ValueError("Number of layers p must be a positive integer.")
        
        self.p = p

        if cost_h is None or mixer_h is None:
            raise ValueError("Cost and mixer Hamiltonians must be provided to build the QAOA circuit.")

        if not isinstance(cost_h, Hamiltonian) or not isinstance(mixer_h, Hamiltonian):
            raise TypeError("Cost and mixer Hamiltonians must be instances of Hamiltonian.")  

        self.mixer_h = mixer_h
        self.cost_h = cost_h

    def build_qnode(self, device: Device, return_counts: bool = False):
        """
        Build the QAOA qnode for expectation value or counts measurement.
        Args:
            device (Device): The device on which to run the QAOA circuit.
            return_counts (bool): Whether to return counts instead of expectation value.
        Returns:
            expval_qnode: A QNode that computes the expectation value or counts.
        Raises:
            ValueError: If the device is not provided.
            TypeError: If the device is not an instance of Device.
        """

        if device is None:
            raise ValueError("A device must be provided to build the QAOA qnode.")
        if not isinstance(device, Device):
            raise TypeError("The device must be an instance of pennylane.devices.Device.")
        
        @qnode(device)
        def expval_qnode(params):
            self._qaoa_circuit(params, wires=device.wires)
            
            if return_counts:
                return counts()
            
            return expval(self.cost_h)

        return expval_qnode
    
    def build_correlations_qnode(self, device: Device, G: Graph):
        """
        Build the QAOA qnode for correlation measurements.
        Args:
            device (Device): The device on which to run the QAOA circuit.
            G (Graph): The graph for which to measure correlations.
        Returns:
            correlations_qnode: A QNode that measures correlations for the given graph.
        Raises:
            ValueError: If the device or graph is not provided.
            TypeError: If the device is not an instance of Device or graph is not an instance of Graph.
        """
        
        if device is None:
            raise ValueError("A device must be provided to build the correlations QNode.")
        if not isinstance(device, Device):
            raise TypeError("The device must be an instance of pennylane.devices.Device.")
        if G is None:
            raise ValueError("A graph must be provided to build the correlations QNode.")
        if not isinstance(G, Graph):
            raise TypeError("Graph must be an instance of networkx.Graph.")
        
        @qnode(device)
        def correlations_qnode(params):
            self._qaoa_circuit(params, wires=device.wires)
            one_point_obs = stack([expval(PauliZ(i)) for i in G.nodes()])
            two_point_obs = stack([expval(PauliZ(i) @ PauliZ(j)) for i, j in G.edges()])
            return concatenate([one_point_obs, two_point_obs], axis=0)

        return correlations_qnode
        
    def _qaoa_layer(self, gamma: float, beta: float):
        """
        Build a single layer of the QAOA circuit.
        Args:
            gamma (float): Parameter for the cost layer.
            beta (float): Parameter for the mixer layer.
        Raises:
            TypeError: If gamma or beta are not floats.
        """

        if not isinstance(gamma, float):
            raise TypeError("Gamma must be a float.")
        if not isinstance(beta, float):
            raise TypeError("Beta must be a float.")

        cost_layer(gamma, self.cost_h)
        mixer_layer(beta, self.mixer_h)
    
    def _qaoa_circuit(self, params: ndarray, wires: Wires = None):
        """
        Build the full QAOA circuit with p layers.
        Args:
            params (ndarray): Parameters for the QAOA circuit, should be of length 2*p.
            wires (Wires): Wires on which to apply the QAOA circuit.
        Raises:
            ValueError: If the number of parameters is not equal to 2*p.
        """

        if len(params) != 2 * self.p:
            raise ValueError(f"Expected {2 * self.p} parameters, but got {len(params)}.")
        
        gammas, betas = params[:self.p], params[self.p:]

        for w in wires:
            Hadamard(wires=w)
        layer(self._qaoa_layer, self.p, gammas, betas)
