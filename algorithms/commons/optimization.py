from numpy import ndarray
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, config: dict | None = None):
        defaults = {"method": "COBYLA", "tol": 1e-2, "maxiter": 100}
        cfg = dict(defaults)
        if config:
            cfg.update(config)

        self.method = cfg["method"]
        self.tol = cfg["tol"]
        self.maxiter = cfg["maxiter"]

    def optimize(self, cost_function, init_params: ndarray) -> ndarray:
        res = minimize(cost_function, x0=init_params, method=self.method, tol=self.tol, options={"maxiter": self.maxiter})
        if not res.success:
            raise OptimizationError(f"Optimization failed: {res.message}")
        return res.x

class OptimizationError(Exception):
    """Custom exception for optimization errors."""
    pass