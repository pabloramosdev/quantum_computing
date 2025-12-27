from pennylane import (
    Hadamard,
    PauliZ,
    counts,
    expval,
    qnode,
    device
)

from pennylane.math import concatenate, stack
from pennylane.qaoa import cost_layer, mixer_layer

from ..problems import QAOAProblemMapping

class QAOABuilder:

    def __init__(self, p: int, device_name: str, shots: int):
        self.p = p
        self.device_name = device_name
        self.shots = shots

    def build(self, problem: QAOAProblemMapping, measurement: str = "expval"):

        dev = device(name=self.device_name, wires=problem.nodes(), shots=self.shots)
        
        @qnode(dev)
        def qaoa_qnode(params):
            if len(params) != 2 * self.p:
                raise ValueError(f"Expected {2 * self.p} parameters, but got {len(params)}.")
        
            gammas, betas = params[:self.p], params[self.p:]

            cost_h, mixer_h = problem.cost_and_mixer_hamiltonians()

            wires = problem.nodes()
            for w in wires:
                Hadamard(wires=w)
            for p in range(self.p):
                cost_layer(gammas[p], cost_h)
                mixer_layer(betas[p], mixer_h)

            if measurement == "expval":
                return expval(cost_h)
            elif measurement == "counts":
                return counts(all_outcomes=True)
            elif measurement == "correlations":
                one_point_obs = stack([expval(PauliZ(i)) for i in problem.nodes()])
                two_point_obs = stack([expval(PauliZ(i) @ PauliZ(j)) for i, j in problem.edges()])
                return concatenate([one_point_obs, two_point_obs], axis=0)
            else:
                raise ValueError(f"Unsupported measurement type: {measurement}")

        return qaoa_qnode
