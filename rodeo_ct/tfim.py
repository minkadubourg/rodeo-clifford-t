"""Transverse-field Ising chain and its first-order Trotter step.

    H = -J sum_k Z_k Z_{k+1} + h sum_k X_k        (open chain, n qubits)

One Trotter step of length dt is
    W(dt) = prod_k Rzz_{k,k+1}(-2 J dt) * prod_k Rx_k(2 h dt),
with Rzz(phi) = CNOT (I x Rz(phi)) CNOT and Rx(theta) = H Rz(theta) H. Every
rotation is therefore a Z rotation, which is what gets compiled to Clifford+T.
"""
from __future__ import annotations

from functools import reduce

import numpy as np

from .synthesis import rz, rz_gridsynth, rz_nearest_pi4

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0 + 0j, -1.0])
HAD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def kron(*ops):
    return reduce(np.kron, ops)


def embed(op, site, n, width=1):
    return kron(*([I2] * site + [op] + [I2] * (n - site - width)))


def hamiltonian(n, J=1.0, h=1.0):
    H = np.zeros((2**n, 2**n), dtype=complex)
    for k in range(n - 1):
        H -= J * embed(np.kron(Z, Z), k, n, 2)
    for k in range(n):
        H += h * embed(X, k, n)
    return H


def _rz_compiled(theta, mode, eps):
    """(matrix, T-count) for Rz(theta) under a compilation mode."""
    if mode == "exact":
        return rz(theta), 0
    if mode == "gridsynth":
        U, _, tc = rz_gridsynth(float(theta), float(eps))
        return U, tc
    if mode == "nearest_pi4":
        return rz_nearest_pi4(theta)
    raise ValueError(mode)


def trotter_step(n, dt, J=1.0, h=1.0, mode="exact", eps=1e-6, compile_zz=True):
    """Return (W, T-count per step). mode: 'exact', 'gridsynth' or 'nearest_pi4'.
    compile_zz=False keeps the ZZ rotations exact and compiles only the field term."""
    rzz_z, tc_zz = _rz_compiled(-2 * J * dt, mode if compile_zz else "exact", eps)
    rx_z, tc_x = _rz_compiled(2 * h * dt, mode, eps)
    rzz = CNOT @ np.kron(I2, rzz_z) @ CNOT
    rx = HAD @ rx_z @ HAD
    W = np.eye(2**n, dtype=complex)
    for k in range(n - 1):
        W = embed(rzz, k, n, 2) @ W
    for k in range(n):
        W = embed(rx, k, n) @ W
    return W, (n - 1) * tc_zz + n * tc_x


def effective_energies(W, dt):
    """Energies of the effective Hamiltonian W = exp(-i H_eff dt), sorted."""
    phases = np.angle(np.linalg.eigvals(W))
    return np.sort(-phases / dt)
