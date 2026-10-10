"""Clifford+T compilation of single-qubit Z rotations.

* `rz_gridsynth`: Ross-Selinger approximation (via the pygridsynth package) to
  operator-norm precision eps, re-synthesised exactly with exact_synthesis.py so
  that the returned word is verified and the global phase is exact.
* `rz_nearest_pi4`: the naive alternative of rounding the angle to the nearest
  multiple of pi/4, i.e. searching only over diagonal Clifford+T gates. It is
  included as a baseline because it silently maps every small angle to identity.
"""
from __future__ import annotations

from functools import lru_cache

import mpmath
import numpy as np
import pygridsynth

from .exact_synthesis import synthesize
from .ring import t_count, word_to_matrix


def rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


@lru_cache(maxsize=None)
def rz_gridsynth(theta: float, eps: float):
    """Return (matrix, word, T-count) with ||matrix - Rz(theta)|| <= eps."""
    word = pygridsynth.gridsynth_gates(theta=mpmath.mpf(repr(theta)), epsilon=mpmath.mpf(repr(eps)))
    exact = word_to_matrix(word)
    ours = synthesize(exact)          # independent exact re-synthesis
    assert word_to_matrix(ours) == exact
    return exact.to_numpy(), ours, t_count(ours)


def rz_nearest_pi4(theta: float):
    """Diagonal Clifford+T gate closest to Rz(theta): angle rounded to a multiple of pi/4."""
    m = int(np.round(theta / (np.pi / 4)))
    return rz(m * np.pi / 4), m % 2  # T-count is 1 for odd multiples of pi/4


def operator_error(U: np.ndarray, V: np.ndarray) -> float:
    return float(np.linalg.norm(U - V, 2))
