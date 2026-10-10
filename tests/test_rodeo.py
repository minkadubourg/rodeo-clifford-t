import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from rodeo_ct import rodeo, tfim


def test_trotter_converges_to_exact_spectrum():
    E = np.linalg.eigvalsh(tfim.hamiltonian(3))
    for dt, tol in ((0.1, 2e-2), (0.01, 2e-4)):
        W, _ = tfim.trotter_step(3, dt)
        assert np.max(np.abs(tfim.effective_energies(W, dt) - E)) < tol


def test_two_spin_spectrum_is_pm1_pm_sqrt5():
    assert np.allclose(np.linalg.eigvalsh(tfim.hamiltonian(2)), [-np.sqrt(5), -1, 1, np.sqrt(5)])


def _circuit_success_probability(W, psi0, steps, dt, E):
    """Noise-free statevector simulation of the Rodeo circuit (one ancilla per cycle)."""
    n_anc, n_obj = len(steps), int(np.log2(W.shape[0]))
    qc = QuantumCircuit(n_anc + n_obj)
    qc.initialize(psi0, list(range(n_anc, n_anc + n_obj)))
    for a, n_k in enumerate(steps):
        qc.x(a)
        qc.h(a)
        U = UnitaryGate(np.linalg.matrix_power(W, int(n_k)))
        qc.append(U.control(1), [a] + list(range(n_anc, n_anc + n_obj)))
        qc.p(E * n_k * dt, a)
        qc.h(a)
    probs = Statevector(qc).probabilities_dict(qargs=list(range(n_anc)))
    return probs.get("1" * n_anc, 0.0)


@pytest.mark.parametrize("mode", ["exact", "gridsynth"])
def test_analytic_filter_matches_circuit_simulation(mode):
    dt = 0.1
    W, _ = tfim.trotter_step(2, dt, mode=mode, eps=1e-3)
    psi = np.kron([1, 0], [1, 0]).astype(complex)
    steps = np.array([[7, 23, 41]])
    for E in (-2.2, -1.0, 0.3, 2.24):
        analytic = rodeo.success_probability(W, dt, psi, [E], steps)[0]
        circuit = _circuit_success_probability(W, psi, steps[0], dt, E)
        assert analytic == pytest.approx(circuit, abs=1e-9)
