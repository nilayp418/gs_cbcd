"""Python side of fast_scf.c: integrals and guess in Python, everything else in one C call.

    sysm = build("H2O")                    # h, Q, C0 as in pyscf_wrapper.ipynb, plus E_ref
    E, wall, t_fock, outer, inner, t_inner = run("cbcd", sysm)

or as drop-ins for the Python drivers (same calls and returns; see the end of this file):

    from code_in_c.fast_scf import *   # pure_scf, cg_scf, cbcd_scf

Integrals are exactly the notebook's: oracles.build_integrals (Lowdin basis, exact ERIs) and
Q = 2 (pq|rs) - (pr|qs). C0 = S^{1/2} C_MINAO[:, :N_o]. E_ref = PySCF RHF energy minus E_nuc.
"""

import ctypes
import os
import subprocess
import sys
import time

import numpy as np
import scipy as sp
from pyscf import df, gto, lib, lo, scf

HERE = os.path.dirname(os.path.abspath(__file__))
GS = os.path.abspath(os.path.join(HERE, ".."))
REF = os.path.abspath(os.path.join(GS, "..", "ref_data"))
sys.path.insert(0, GS)
from oracles import build_integrals  # noqa: E402

__all__ = ["pure_scf", "cg_scf", "cbcd_scf"]                    # import *: just the drop-ins (end of file)

def _load(name, deps):
    """Compile <name>.c into lib<name>.dylib if it is missing or older than its sources."""
    lib, src = os.path.join(HERE, f"lib{name}.dylib"), [os.path.join(HERE, f) for f in deps]
    if not os.path.exists(lib) or os.path.getmtime(lib) < max(map(os.path.getmtime, src)):
        subprocess.run(["clang", "-O3", "-mcpu=native", "-shared", "-fPIC", "-o", lib, src[0],
                        "-framework", "Accelerate"], check=True)
    return ctypes.CDLL(lib)


_lib = _load("fast_scf", ["fast_scf.c"])                       # SCF, CG, CBCD
_I, _D, _U = ctypes.c_int, ctypes.c_double, ctypes.c_uint64
_A = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_F = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
_lib.fast_scf.argtypes = [_I, _I, _I, _U, _I, _I, _A, _A, _F, _D, _I, _D, _D, _A]
_lib.fast_scf_df.argtypes = [_I, _I, _I, _U, _I, _I, _I, _A, _A, _F, _D, _I, _D, _D, _A]
for f in (_lib.fast_scf, _lib.fast_scf_df):
    f.restype = _I
_lib.prof_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]  # energy profiles
_lib.prof_set.restype = None                                                                   # (opt_profiling)
_lib.prof_count.restype = ctypes.c_long
PROF_CAP = 2_000_000                                             # max inner energies recorded per run


METHODS = {"scf": 0, "cg": 1, "cbcd": 2}                                       # fast_scf.c
BASIS, EPS, ITER_MAX, DAMP = "aug-cc-pVTZ", 1e-6, 10000, 0.4
# matched inner stopping rule: CG stops when one iteration lowers E by <= INNER_TOL["cg"];
# CBCD c = 0, 1, 2 when its last N_o steps (one CG iteration's worth of column updates) together
# do (rolling window); CBCD c = N_o (Gauss-Southwell) when one step lowers E by <= INNER_TOL["cbcd"] / N_o
# (a step on the steepest column bounds the others, so N_o such steps total <= INNER_TOL["cbcd"]).
INNER_TOL = {"scf": 0.0, "cg": 1e-8, "cbcd": 1e-8}


def build(name, basis=BASIS):
    """Integrals + guess + reference energy (all untimed Python)."""
    t0 = time.perf_counter()
    with open(os.path.join(REF, f"{name}.txt")) as fh:
        atom = "".join(fh.readlines()[1:])
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    integ = build_integrals(mol)
    nb, no = mol.nao, mol.nelectron // 2
    Q = 2 * integ.eri
    Q -= integ.eri.transpose(0, 2, 1, 3)               # Q = 2 eri - einsum("prqs->pqrs", eri)
    h = np.ascontiguousarray(integ.h)
    del integ
    g = scf.RHF(mol); g.max_cycle = 0; g.kernel()
    C0 = np.real(sp.linalg.fractional_matrix_power(g.get_ovlp(), 0.5)) @ g.mo_coeff[:, :no]
    e_ref = scf.RHF(mol).kernel() - mol.get_enuc()
    return dict(name=name, nb=nb, no=no, ne=mol.nelectron, h=h, Q=Q.reshape(nb * nb, nb * nb),
                C0=np.asfortranarray(C0), e_ref=e_ref, t_build=time.perf_counter() - t0)


def build_df(name, atom, basis="6-31g*", auxbasis=None):
    """Density-fitted system (all untimed Python): Lowdin-basis h, DF tensor B (naux x nb x nb,
    (pq|rs) ~ sum_P B_P[pq] B_P[rs], from PySCF's Cholesky-decomposed 3-center integrals with its
    default JK-fit auxiliary basis), MINAO guess, and the PySCF DF-RHF reference energy (same
    auxiliary basis), so the error measures convergence, not the DF approximation."""
    t0 = time.perf_counter()
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    auxbasis = auxbasis or df.make_auxbasis(mol)
    nb, no = mol.nao, mol.nelectron // 2
    X = lo.orth.lowdin(mol.intor("int1e_ovlp"))
    h = np.ascontiguousarray(X.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ X)
    cderi = df.incore.cholesky_eri(mol, auxbasis=auxbasis)          # naux x nb(nb+1)/2
    naux = cderi.shape[0]
    B = np.empty((naux, nb, nb))
    for P0 in range(0, naux, 128):                                  # B_P = X^T L_P X, in blocks
        L = lib.unpack_tril(cderi[P0:P0 + 128])
        B[P0:P0 + 128] = X.T @ L @ X
    ref = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    ref.with_df._cderi = cderi                                      # same DF integrals, no second copy
    e_ref = ref.kernel() - mol.get_enuc()
    del cderi, ref
    g = scf.RHF(mol); g.max_cycle = 0; g.kernel()
    C0 = np.real(sp.linalg.fractional_matrix_power(g.get_ovlp(), 0.5)) @ g.mo_coeff[:, :no]
    return dict(name=name, nb=nb, no=no, naux=naux, ne=mol.nelectron, h=h, B=B, C0=np.asfortranarray(C0),
                e_ref=e_ref, t_build=time.perf_counter() - t0)


def run(method, s, c=None, seed=None, eps=EPS, damp=DAMP, tol=None, window=False, opt_profiling=False,
        iter_max=ITER_MAX):
    """One full SCF in C. method: "scf" | "cg" | "cbcd" (c = 0, 1, 2, ... or None for N_o).
    Returns (E, wall, t_fock, outer, inner, t_inner), times measured in C.
    opt_profiling=True also returns (scf_profile, sub_iter_energies) as the Python drivers do:
    E after the guess and each outer iteration, and per outer iteration the inner objective
    Tr(C^T F C) at the start and after every CG iteration / CBCD step (empty lists for SCF).
    Use separate runs for timing."""
    _, out, prof = _solve(method, s, c, seed, eps, damp, tol, window, opt_profiling, iter_max)
    res = out[0], out[1], out[2], int(out[3]), int(out[4]), out[5]
    return (res, prof) if opt_profiling else res


def _solve(method, s, c, seed, eps, damp, tol, window, opt_profiling, iter_max):
    """run's C call: (final C, out, profile or None)."""
    if opt_profiling:
        po, pi, pl = np.zeros(iter_max + 2), np.zeros(PROF_CAP), np.zeros(iter_max + 2, dtype=np.int64)
        _lib.prof_set(po.ctypes.data, pi.ctypes.data, pl.ctypes.data, PROF_CAP)
        try:
            C, out, _ = _solve(method, s, c, seed, eps, damp, tol, window, False, iter_max)
            n = _lib.prof_count()
        finally:
            _lib.prof_set(None, None, None, 0)                  # profiling off again
        if n > PROF_CAP:
            raise RuntimeError(f"profile overflow: {n} inner entries > PROF_CAP = {PROF_CAP}")
        it = int(out[3])
        subs = np.split(pi[:n], np.cumsum(pl[:it])[:-1]) if it else []
        return C, out, (po[:it + 1].copy(), [x.copy() for x in subs])
    no = s["no"]
    c = no if c is None else c
    if tol is None:   # c = N_o per step: tol / N_o; with window=True it uses the c = 0, 1, 2 window rule
        tol = INNER_TOL[method] / (no if method == "cbcd" and c == no and not window else 1)
    seed = int.from_bytes(os.urandom(8), "little") if seed is None else seed
    C = s["C0"].copy(order="F")
    out = np.zeros(6)
    if "B" in s:
        rc = _lib.fast_scf_df(METHODS[method], c, int(window), seed, s["nb"], no, s["naux"], s["h"], s["B"], C,
                              eps, iter_max, damp, tol, out)
    else:
        rc = _lib.fast_scf(METHODS[method], c, int(window), seed, s["nb"], no, s["h"], s["Q"], C, eps, iter_max,
                           damp, tol, out)
    if rc != 0:
        raise RuntimeError(f"{method} did not converge on {s['name']}")
    return C, out, None


# ---------------------------------------------------------------- drop-ins for python_impls/{pure,cg,cbcd}_scf.py
# Same signatures and return shapes as the Python drivers, so `from code_in_c.fast_scf import *` after
# the python_impls imports switches a notebook to C. The whole SCF runs in C on the oracle's h and Q
# (make_rhf_oracle_energy); f.F is not updated. Differences from the Python drivers:
#   - inner loops stop by fast_scf.c's rules (INNER_TOL; see the header of fast_scf.c), so CBCD inner step
#     counts differ from circle_optimize's
#   - c_value = 0 draws its column orders from fast_scf.c's own RNG (seed=None: random; pass seed to repeat)
#   - timing=True with opt_profiling=True runs the SCF twice (same seed): once timed, once recording energies
#   - grad_f, expressive and fast_proj are accepted and ignored (C always uses the implicit projection)
#   - non-convergence prints the drivers' message and returns None, as they do


def _drop_in(method, f, C0, c, timing, eps, ITER_MAX, damp_param, opt_profiling, seed, idx_set, pqr=True):
    if not (isinstance(idx_set, slice) and idx_set == slice(None)):
        raise NotImplementedError("fast_scf: idx_set (frozen columns) is not supported")
    if not pqr:
        raise NotImplementedError("fast_scf: pqr=False is not supported (C always uses pivoted QR)")
    nb = f.n_b
    s = dict(name=f"{method} drop-in", nb=nb, no=C0.shape[1], h=np.ascontiguousarray(f.h, dtype=np.float64),
             Q=np.ascontiguousarray(f.Q, dtype=np.float64).reshape(nb * nb, nb * nb),  # a view, no copy of Q
             C0=np.asfortranarray(C0, dtype=np.float64))
    seed = int.from_bytes(os.urandom(8), "little") if seed is None else seed
    args = (method, s, c, seed, eps, damp_param, None, False)
    try:
        if timing or not opt_profiling:
            C, out, _ = _solve(*args, False, ITER_MAX)
        if opt_profiling:
            C_p, out_p, (scf_profile, subs) = _solve(*args, True, ITER_MAX)
            if not timing:
                C, out = C_p, out_p
    except RuntimeError:
        print("SCF failed to converge in {} iterations.".format(ITER_MAX))
        return
    ret = (C, out[0])
    if timing:
        ret += ((out[1], out[2]),)                              # (wall, Fock-build time), as (time, e_time)
    if opt_profiling:
        scf_profile = scf_profile.tolist()
        ret += (scf_profile if method == "scf" else (scf_profile, [x.tolist() for x in subs]),)
    return ret


def pure_scf(f, C0, timing=False, eps=1e-6, ITER_MAX=10000, expressive=True, damp_param=0.4,
             opt_profiling=False):
    return _drop_in("scf", f, C0, None, timing, eps, ITER_MAX, damp_param, opt_profiling, 0, slice(None))


def cg_scf(f, grad_f, C0, timing=False, eps=1e-6, expressive=False, ITER_MAX=10000, damp_param=0.4,
           opt_profiling=False, idx_set=slice(None)):
    return _drop_in("cg", f, C0, None, timing, eps, ITER_MAX, damp_param, opt_profiling, 0, idx_set)


def cbcd_scf(f, grad_f, C0, c_value=1, timing=False, eps=1e-6, ITER_MAX=10000, expressive=False, damp_param=0.4,
             opt_profiling=False, fast_proj=True, pqr=True, idx_set=slice(None), seed=None):
    return _drop_in("cbcd", f, C0, c_value, timing, eps, ITER_MAX, damp_param, opt_profiling, seed, idx_set, pqr)

