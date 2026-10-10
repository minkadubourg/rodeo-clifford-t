"""Rodeo algorithm (Choi, Lee, Bonitati, Qian, Watkins, PRL 127, 040505, 2021).

Each cycle prepares an ancilla in (|0> - |1>)/sqrt(2), applies the evolution
U(t_k) controlled on the ancilla, a phase exp(i E t_k), a Hadamard, and keeps the
run if the ancilla reads 1. For an eigenvector of U(t) with eigenvalue
exp(-i lambda t) the success probability is cos^2((E - lambda) t / 2), so with N
cycles and random times t_k the success probability is an energy filter

    P(E) = sum_j |<v_j|psi>|^2  prod_k cos^2((E - lambda_j) t_k / 2).

Here U(t_k) = W^{n_k} is n_k repetitions of a (possibly compiled) Trotter step W,
so P(E) is evaluated exactly from the eigendecomposition of W. This is
equivalent to a noise-free statevector simulation of the circuit (verified
against Qiskit Aer in tests/test_rodeo_circuit.py) and orders of magnitude faster.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import schur


def eigen_unitary(W):
    """Orthonormal eigenvectors and eigenphases of a unitary (complex Schur form)."""
    T, Zs = schur(W, output="complex")
    return Zs, np.angle(np.diag(T))


def sample_steps(n_cycles, t_rms, dt, n_draws, rng):
    """Integer step counts n_k = round(|t_k| / dt), t_k ~ N(0, t_rms)."""
    t = np.abs(rng.normal(0.0, t_rms, size=(n_draws, n_cycles)))
    return np.maximum(1, np.round(t / dt)).astype(int)


def success_probability(W, dt, psi0, energies, steps):
    """Mean success probability over draws. steps: (n_draws, n_cycles) integers."""
    V, phi = eigen_unitary(W)
    weights = np.abs(V.conj().T @ psi0) ** 2              # (n_eig,)
    times = steps * dt                                    # (n_draws, n_cycles)
    # accumulated eigenphase after n_k steps: lambda_j t_k = -phi_j n_k
    E = np.asarray(energies)[:, None, None, None]
    arg = (E * times[None, :, :, None] + phi[None, None, None, :] * steps[None, :, :, None]) / 2
    filt = np.prod(np.cos(arg) ** 2, axis=2)             # (n_E, n_draws, n_eig)
    return (filt * weights[None, None, :]).sum(-1).mean(-1)
