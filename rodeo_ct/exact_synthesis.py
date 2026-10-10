"""Exact synthesis of single-qubit Clifford+T unitaries (Kliuchnikov, Maslov, Mosca 2013).

For U with entries in Z[1/sqrt(2), i], the denominator exponent s = sde(|U_00|^2)
can be reduced by exactly one with a step U -> H T^k U for some k, as long as
s >= 4. Repeating this "peeling" leaves a unitary with s <= 3, which is looked up
in a finite table built once by breadth-first search. All arithmetic is exact
(see ring.py), so the decomposition reproduces U exactly, global phase included.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache

from .ring import GATES, Mat2, word_to_matrix

H = GATES["H"]


def _t_power(m: int) -> Mat2:
    return word_to_matrix("T" * (m % 8))


_T_POW = [_t_power(m) for m in range(8)]


def _sde(U: Mat2) -> int:
    return U.e[0][0].sde_abs2()


@lru_cache(maxsize=1)
def base_table(max_sde: int = 3, search_sde: int = 4) -> dict:
    """Shortest words for every unitary with sde(|U_00|^2) <= max_sde.

    BFS over the generators {H, T, S, W} from the identity, exploring only nodes
    with sde <= search_sde so the search stays finite."""
    gens = ["H", "T", "S", "W"]
    start = GATES["I"]
    table = {start: ""}
    queue = deque([start])
    while queue:
        U = queue.popleft()
        w = table[U]
        for g in gens:
            V = U @ GATES[g]
            if V in table or _sde(V) > search_sde:
                continue
            table[V] = w + g
            queue.append(V)
    return {U: w for U, w in table.items() if _sde(U) <= max_sde}


def synthesize(U: Mat2) -> str:
    """Gate word over {H, T, S, W} whose exact product (left to right) equals U."""
    prefix = []
    s = _sde(U)
    while s > 3:
        for k in range(8):
            V = H @ _T_POW[k] @ U
            if _sde(V) == s - 1:
                # U = T^{-k} H V
                prefix.append("T" * ((8 - k) % 8) + "H")
                U, s = V, s - 1
                break
        else:
            raise ValueError("sde reduction failed: U is not a Clifford+T unitary")
    table = base_table()
    if U not in table:
        raise KeyError("base case not found in lookup table")
    return "".join(prefix) + table[U]


def simplify(word: str) -> str:
    """Merge runs of T into canonical powers (T^2 -> S, T^8 -> identity) and
    cancel adjacent H pairs, without changing the operator."""
    out = []
    for g in word:
        out.append(g)
        changed = True
        while changed:
            changed = False
            if len(out) >= 2 and out[-1] == out[-2] == "H":
                out.pop(); out.pop(); changed = True
            elif len(out) >= 2 and out[-1] == out[-2] == "T":
                out.pop(); out.pop(); out.append("S"); changed = True
            elif len(out) >= 4 and out[-4:] == ["S"] * 4:
                del out[-4:]; changed = True
            elif len(out) >= 8 and out[-8:] == ["W"] * 8:
                del out[-8:]; changed = True
    return "".join(out)
