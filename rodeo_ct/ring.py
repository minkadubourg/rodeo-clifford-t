"""Exact arithmetic in the ring D[omega] = Z[1/sqrt(2), i], omega = exp(i pi/4).

Every Clifford+T unitary has entries in this ring, so representing them exactly
(rather than as floats) makes exact synthesis a purely integer computation: no
tolerance, no rounding, and the global phase is tracked exactly.

An element is stored as (a + b w + c w^2 + d w^3) / sqrt(2)^k with integer
a, b, c, d and k >= 0, reduced so that k is minimal (the denominator exponent).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_OMEGA = np.exp(1j * np.pi / 4)


def _mul_num(x, y):
    """Product of two numerators in Z[w], using w^4 = -1."""
    out = [0, 0, 0, 0]
    for i, xi in enumerate(x):
        if xi == 0:
            continue
        for j, yj in enumerate(y):
            if yj == 0:
                continue
            p = i + j
            if p < 4:
                out[p] += xi * yj
            else:
                out[p - 4] -= xi * yj
    return tuple(out)


_SQRT2 = (0, 1, 0, -1)          # sqrt(2) = w - w^3


def _div_sqrt2(x):
    """x / sqrt(2) if it lies in Z[w], else None. Uses x / sqrt(2) = x (w - w^3) / 2."""
    y = _mul_num(x, _SQRT2)
    if all(v % 2 == 0 for v in y):
        return tuple(v // 2 for v in y)
    return None


@dataclass(frozen=True)
class DOmega:
    num: tuple  # (a, b, c, d)
    k: int = 0

    @staticmethod
    def make(num, k=0) -> "DOmega":
        num = tuple(int(v) for v in num)
        if not any(num):
            return DOmega((0, 0, 0, 0), 0)
        while k > 0:
            red = _div_sqrt2(num)
            if red is None:
                break
            num, k = red, k - 1
        return DOmega(num, k)

    # ---- constructors -------------------------------------------------------
    @staticmethod
    def integer(n) -> "DOmega":
        return DOmega.make((n, 0, 0, 0))

    @staticmethod
    def omega_pow(m) -> "DOmega":
        m %= 8
        sign = -1 if m >= 4 else 1
        num = [0, 0, 0, 0]
        num[m % 4] = sign
        return DOmega.make(num)

    # ---- arithmetic ---------------------------------------------------------
    def _lift(self, k):
        num = self.num
        for _ in range(k - self.k):
            num = _mul_num(num, _SQRT2)
        return num

    def __add__(self, o: "DOmega") -> "DOmega":
        k = max(self.k, o.k)
        a, b = self._lift(k), o._lift(k)
        return DOmega.make(tuple(x + y for x, y in zip(a, b)), k)

    def __neg__(self) -> "DOmega":
        return DOmega(tuple(-v for v in self.num), self.k)

    def __sub__(self, o: "DOmega") -> "DOmega":
        return self + (-o)

    def __mul__(self, o: "DOmega") -> "DOmega":
        return DOmega.make(_mul_num(self.num, o.num), self.k + o.k)

    def conj(self) -> "DOmega":
        a, b, c, d = self.num  # w* = -w^3, (w^2)* = -w^2, (w^3)* = -w
        return DOmega((a, -d, -c, -b), self.k)

    def is_zero(self) -> bool:
        return not any(self.num)

    def __complex__(self) -> complex:
        a, b, c, d = self.num
        return complex((a + b * _OMEGA + c * _OMEGA**2 + d * _OMEGA**3) / np.sqrt(2) ** self.k)

    def sde_abs2(self) -> int:
        """Denominator exponent of |x|^2, the quantity peeled off by exact synthesis."""
        return (self * self.conj()).k


class Mat2:
    """2x2 matrix over D[omega]."""

    __slots__ = ("e",)

    def __init__(self, e):
        self.e = tuple(tuple(r) for r in e)

    def __matmul__(self, o: "Mat2") -> "Mat2":
        a, b = self.e, o.e
        return Mat2([[a[i][0] * b[0][j] + a[i][1] * b[1][j] for j in range(2)] for i in range(2)])

    def __eq__(self, o) -> bool:
        return isinstance(o, Mat2) and self.e == o.e

    def __hash__(self):
        return hash(self.e)

    def to_numpy(self) -> np.ndarray:
        return np.array([[complex(x) for x in r] for r in self.e])

    def times_scalar(self, s: DOmega) -> "Mat2":
        return Mat2([[x * s for x in r] for r in self.e])


Z0, ONE = DOmega.integer(0), DOmega.integer(1)
INV_SQRT2 = DOmega.make((1, 0, 0, 0), 1)

GATES = {
    "I": Mat2([[ONE, Z0], [Z0, ONE]]),
    "H": Mat2([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]]),
    "T": Mat2([[ONE, Z0], [Z0, DOmega.omega_pow(1)]]),
    "S": Mat2([[ONE, Z0], [Z0, DOmega.omega_pow(2)]]),
    "X": Mat2([[Z0, ONE], [ONE, Z0]]),
    "W": Mat2([[DOmega.omega_pow(1), Z0], [Z0, DOmega.omega_pow(1)]]),  # global phase w
}


def word_to_matrix(word: str) -> Mat2:
    """Exact operator of a gate word, read as a matrix product from left to right."""
    m = GATES["I"]
    for g in word:
        m = m @ GATES[g]
    return m


def t_count(word: str) -> int:
    """Number of T gates after merging runs (T^2 = S is Clifford)."""
    count, run = 0, 0
    for g in word + "|":
        if g == "T":
            run += 1
        else:
            count += run % 2
            run = 0
    return count
