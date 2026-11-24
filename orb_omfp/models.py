"""Bias-free ReLU kernels and label-centered ridge regression."""

import numpy as np
from scipy.optimize import minimize_scalar
from .config import U


def arccos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    na, nb = (np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1))
    c = np.clip(A @ B.T / np.outer(na, nb), -1.0, 1.0)
    th = np.arccos(c)
    return np.outer(na, nb) * (np.sin(th) + (np.pi - th) * c) / (2 * np.pi)


class Model:
    def __init__(self, X, O, y):
        self.X, self.O, self.y = (X, O, y)
        self.n, self.no = (len(X), len(O))
        self.G = arccos(X, X)
        self.C = arccos(O, X)
        self.m = 0.5
        # Reuse one eigendecomposition across the penalty grid.
        ev, self.V = np.linalg.eigh(self.G)
        self.ev = np.maximum(ev, 0.0)
        self.CV = self.C @ self.V
        self.mean = float(y.mean())
        self.b = self.V.T @ (y - self.mean)

    def inverse(self, u):
        a = np.atleast_1d(u)
        w = 1 / (self.ev[:, None] + self.m * np.maximum(a[None, :], 1e-300))
        for j, v in enumerate(a):
            if v == 0:
                cutoff = 1e-12 * self.ev.max()
                w[:, j] = np.where(self.ev > cutoff, 1 / np.maximum(self.ev, cutoff), 0.0)
            elif np.isinf(v):
                w[:, j] = 0.0
        return w

    def predict(self, u):
        return self.mean + (self.CV @ (self.inverse(u) * self.b[:, None])).T

    def operator(self, u):
        return self.CV * self.inverse(u).ravel() @ self.V.T

    def loss(self, u, yo):
        return np.mean((self.predict(u) - yo) ** 2, axis=1)

    def best(self, yo):
        vals = self.loss(U, yo)
        j = int(vals.argmin())
        best = float(U[j])
        if 0 < j < len(U) - 2:
            res = minimize_scalar(
                lambda v: float(self.loss(np.exp(v), yo)[0]),
                bounds=np.log(U[[j - 1, j + 1]]),
                method="bounded",
            )
            best = float(np.exp(res.x))
        return best
