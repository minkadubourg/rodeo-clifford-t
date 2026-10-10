"""Run every experiment in the README and save results/*.npz.

1. spectra   Rodeo spectra for four initial states: exact evolution, Clifford+T
             (gridsynth) evolution, and the nearest-pi/4 baseline.
2. budget    Effective-Hamiltonian energy error vs Trotter step dt and synthesis
             precision eps, for n = 2 and n = 4 spins.
3. pitfall   Rodeo spectrum when small steps are combined with a loose eps.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rodeo_ct import rodeo, tfim  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
J = h = 1.0
STATES = {
    "|++>": np.kron([1, 1], [1, 1]) / 2,
    "|+->": np.kron([1, 1], [1, -1]) / 2,
    "|00>": np.kron([1, 0], [1, 0]).astype(float),
    "|01>": np.kron([1, 0], [0, 1]).astype(float),
}
N_CYCLES, T_RMS, N_DRAWS = 10, 5.0, 400
ENERGIES = np.linspace(-3, 3, 1201)


def spectra():
    dt = 0.1
    rng = np.random.default_rng(1)
    steps = rodeo.sample_steps(N_CYCLES, T_RMS, dt, N_DRAWS, rng)
    H = tfim.hamiltonian(2, J, h)
    evals, evecs = np.linalg.eigh(H)
    cases = {
        "exact": tfim.trotter_step(2, dt, J, h, "exact")[0],
        "gridsynth_1e-3": tfim.trotter_step(2, dt, J, h, "gridsynth", 1e-3)[0],
        "nearest_pi4_field": tfim.trotter_step(2, dt, J, h, "nearest_pi4", compile_zz=False)[0],
    }
    out = {"energies": ENERGIES, "evals": evals}
    for sname, psi in STATES.items():
        out[f"weights {sname}"] = np.abs(evecs.conj().T @ psi) ** 2
        for cname, W in cases.items():
            out[f"{cname} {sname}"] = rodeo.success_probability(W, dt, psi, ENERGIES, steps)
    np.savez(os.path.join(OUT, "spectra.npz"), **out)


def budget():
    dts = np.logspace(-3, np.log10(0.5), 26)
    epss = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    out = {"dts": dts, "epss": np.array(epss)}
    for n in (2, 4):
        exact_E = np.linalg.eigvalsh(tfim.hamiltonian(n, J, h))
        err = np.zeros((len(epss) + 1, len(dts)))
        tcount = np.zeros((len(epss), len(dts)))
        for j, dt in enumerate(dts):
            W, _ = tfim.trotter_step(n, dt, J, h, "exact")
            err[0, j] = np.max(np.abs(tfim.effective_energies(W, dt) - exact_E))
            for i, eps in enumerate(epss):
                W, tc = tfim.trotter_step(n, dt, J, h, "gridsynth", eps)
                err[i + 1, j] = np.max(np.abs(tfim.effective_energies(W, dt) - exact_E))
                tcount[i, j] = tc
        out[f"err_n{n}"], out[f"tcount_n{n}"] = err, tcount
        print(f"budget n={n} done")
    np.savez(os.path.join(OUT, "budget.npz"), **out)


def pitfall():
    rng = np.random.default_rng(2)
    psi = STATES["|00>"]
    out = {"energies": ENERGIES}
    for label, dt, eps in (("dt=0.1, eps=1e-3", 0.1, 1e-3), ("dt=0.005, eps=1e-3", 0.005, 1e-3),
                           ("dt=0.005, exact", 0.005, None)):
        steps = rodeo.sample_steps(N_CYCLES, T_RMS, dt, N_DRAWS, rng)
        W = tfim.trotter_step(2, dt, J, h, "exact" if eps is None else "gridsynth", eps or 1e-6)[0]
        out[label] = rodeo.success_probability(W, dt, psi, ENERGIES, steps)
    np.savez(os.path.join(OUT, "pitfall.npz"), **out)


if __name__ == "__main__":
    for f in (spectra, pitfall, budget):
        t0 = time.time()
        f()
        print(f"{f.__name__}: {time.time() - t0:.1f} s")
