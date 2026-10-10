import random

import numpy as np
import pytest

from rodeo_ct.exact_synthesis import synthesize
from rodeo_ct.ring import DOmega, t_count, word_to_matrix
from rodeo_ct.synthesis import operator_error, rz, rz_gridsynth, rz_nearest_pi4


def test_ring_matches_complex_arithmetic():
    rng = random.Random(0)
    for _ in range(2000):
        a = DOmega.make([rng.randint(-6, 6) for _ in range(4)], rng.randint(0, 5))
        b = DOmega.make([rng.randint(-6, 6) for _ in range(4)], rng.randint(0, 5))
        assert abs(complex(a * b) - complex(a) * complex(b)) < 1e-9
        assert abs(complex(a + b) - (complex(a) + complex(b))) < 1e-9
        assert abs(complex(a.conj()) - complex(a).conjugate()) < 1e-9


def test_exact_synthesis_round_trip_random_words():
    rng = random.Random(1)
    for _ in range(300):
        word = "".join(rng.choice("HTS") for _ in range(rng.randint(1, 80)))
        U = word_to_matrix(word)
        assert word_to_matrix(synthesize(U)) == U          # exact equality, phase included


@pytest.mark.parametrize("theta,eps", [(0.02, 1e-3), (1.3, 1e-6), (-0.7, 1e-8), (2.5, 1e-10)])
def test_gridsynth_error_bound_and_optimal_t_count(theta, eps):
    U, word, tc = rz_gridsynth(theta, eps)
    assert operator_error(U, rz(theta)) <= eps                 # not up to phase: exact phase kept
    assert np.allclose(word_to_matrix(word).to_numpy(), U)
    assert tc <= 3 * np.log2(1 / eps) + 12                     # Ross-Selinger scaling


def test_nearest_pi4_maps_small_angles_to_identity():
    for theta in (0.002, 0.02, 0.2, 0.39):
        U, _ = rz_nearest_pi4(theta)
        assert np.allclose(U, np.eye(2))


def test_t_count_merges_runs():
    assert t_count("TT") == 0 and t_count("TTT") == 1 and t_count("THT") == 2
