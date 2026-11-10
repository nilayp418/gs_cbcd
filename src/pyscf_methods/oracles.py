from dataclasses import dataclass

import numpy as np
from pyscf import gto, lo, ao2mo
from math import sin, cos, atan
from numpy import dot, real
from line_profiler import profile
from scipy.linalg.blas import dger


@dataclass
class Integrals:
    h: np.ndarray
    eri: np.ndarray
    n_b: int = 0


@dataclass
class RHFOracleEnergy:
    n_b: int
    n_e: int
    F: np.ndarray
    h: np.ndarray
    Q: np.ndarray
    fc_scratch: np.ndarray = None  # lazy (n_b, N_o) F-order buffer for scf_style traces
    # fc : np.ndarray
    # fs : np.ndarray


@dataclass
class RHFGradEnergy:
    f: RHFOracleEnergy
    grad_f_mem: np.ndarray
    idx_set: list = None
    g2l: dict = None


@dataclass
class RHFProjGradEnergy:
    grad_f: RHFGradEnergy
    proj_grad: np.ndarray


def build_integrals(mol):
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    X = lo.orth.lowdin(S)
    n_b = np.size(S, 0)
    h = X.T @ h @ X
    eri = ao2mo.restore(1, ao2mo.kernel(mol, X), mol.nao)
    return Integrals(h=h, eri=eri, n_b=n_b)


def make_rhf_oracle_energy(integrals, n_e):
    if not (n_e % 2 == 0):
        raise ValueError("RHF should not be used on open shell systems")
    n_b = integrals.n_b

    h = integrals.h
    Q = 2 * (integrals.eri) - np.einsum("prqs->pqrs", integrals.eri)

    F = h
    # fc = np.zeros(n_b)
    # fs = np.zeros(n_b)
    # return RHFOracleEnergy(n_b, n_e, F, h, Q, fc, fs)
    return RHFOracleEnergy(n_b, n_e, F, h, Q)


def make_rhf_grad_energy(f, idx_set=slice(None)):
    if isinstance(idx_set, slice) and idx_set == slice(None):
        grad_f_mem = np.zeros((f.n_b, f.n_e // 2))
        g2l = None
    else:
        # normalize to a plain int list so every downstream consumer (grad_energy,
        # proj_grad, mi_amore) only ever sees slice(None) or a list of global indices
        if isinstance(idx_set, np.ndarray) and idx_set.dtype == bool:
            idx_set = np.flatnonzero(idx_set).tolist()
        else:
            idx_set = list(idx_set)
        grad_f_mem = np.zeros((f.n_b, len(idx_set)))
        g2l = {g: l for l, g in enumerate(idx_set)}

    return RHFGradEnergy(f, grad_f_mem, idx_set, g2l)


# one-time use, irreversible!
def localize_grad(grf, idx_set):
    if isinstance(idx_set, slice) and idx_set == slice(None):
        return grf
    elif isinstance(idx_set, np.ndarray) and idx_set.dtype == bool:
        idx_set = np.flatnonzero(idx_set).tolist()
    else:
        idx_set = list(idx_set)

    grf.idx_set = idx_set
    grf.g2l = {g: l for l, g in enumerate(idx_set)}
    grf.grad_f_mem = grf.grad_f_mem[:, idx_set]
    return grf


def make_proj_grad_energy(gradf_obj: RHFGradEnergy):
    return RHFProjGradEnergy(
        gradf_obj, np.zeros(np.shape(gradf_obj.grad_f_mem), order="F")
    )


@profile
def proj_grad(pg: RHFProjGradEnergy, C):
    gradf = pg.grad_f.grad_f_mem
    np.subtract(gradf, C @ (C.T @ gradf), out=pg.proj_grad)
    return pg.proj_grad


# def proj_grad_amor(pg : RHFProjGradEnergy, C, pert_vec, pert_ind, t): # assume gradient has already been updated; new entry is C except in pos'n idx
#     ct, st = cos(t), sin(t)
#     fc = pg.grad_f.f.fc
#     old_vec = C[:, pert_ind]
#     eps =  (ct-1) * old_vec + st * pert_vec
#     nc = old_vec + eps
#     feps = 2 * ((ct - 1) * fc + st * pg.grad_f.f.fs)

#     gradf = pg.grad_f.grad_f_mem # 2FC^{m+1}, already updated at col pert_ind

#     Ct_feps = C.T @ feps
#     pg.proj_grad[:, pert_ind] += (feps - C @ (Ct_feps))

#     Ct_feps[pert_ind] = dot(feps, nc) # now is eps.T @ gradf

#     pg.proj_grad -= np.outer(old_vec, Ct_feps) + np.outer(eps, ((nc).T @ gradf))

#     return pg.proj_grad


@profile
def energy(
    f,
    C,
    fock_update_only=False,
    scf_style=False,
    silently_full=False,
    full_and_mix=(False, 0),
):
    if silently_full or full_and_mix[0]:
        old_fock = f.F

    if scf_style:
        if f.fc_scratch is None or f.fc_scratch.shape != C.shape:
            f.fc_scratch = np.empty(C.shape, order="F")
        np.matmul(f.F, C, out=f.fc_scratch)
        return np.dot(f.fc_scratch.ravel(order="F"), C.ravel(order="F"))

    D = C @ C.T
    # single BLAS matvec over the (n_b^2, n_b^2) view of Q -- equivalent to
    # einsum("pqrs,rs->pq", Q, D) without per-call einsum/path overhead
    f.F = f.h + (f.Q.reshape(f.n_b * f.n_b, f.n_b * f.n_b) @ D.ravel()).reshape(
        f.n_b, f.n_b
    )
    f.F = (f.F + np.transpose(f.F)) / 2  # symmetrization!

    if full_and_mix[0]:
        Dr = D.ravel()
        e = np.dot(Dr, f.h.ravel()) + np.dot(Dr, f.F.ravel())
        f.F = f.F * (1 - full_and_mix[1]) + full_and_mix[1] * old_fock
        return e

    if fock_update_only:
        return

    Dr = D.ravel()
    if not (silently_full):
        return np.dot(Dr, f.h.ravel()) + np.dot(Dr, f.F.ravel())

    e = np.dot(Dr, f.h.ravel()) + np.dot(Dr, f.F.ravel())
    f.F[:, :] = old_fock
    return e


# def energy_amor(f, grf, C, pert_vec, pert_ind, f_ref): #returns the minimal t and the corresponding reference energy at the
#     # minimal t value when perturbing the pert_ind column in C in the C[:, pert_ind]*cos + pert_vec * sin way.
#     sin_col = pert_vec
#     f.fc = grf.grad_f_mem[:, pert_ind]/2
#     f.fs = f.F @ sin_col
#     t1 = dot(C[:, pert_ind], f.fc)
#     t2 = dot(sin_col, f.fs)
#     t3 = dot(sin_col, f.fc)

#     tmin = np.arctan(2*t3 / (t1 - t2))/2

#     qty = 2*t3*np.sin(2*tmin) + (t1 - t2)*np.cos(2*tmin)

#     if qty < 0:
#         return tmin, f_ref + real((t2 - t1 + qty)/2)
#     else:
#         return tmin + np.pi/2, f_ref + real((t2 - t1 - qty)/2)


def get_curr_grad(grad_f):
    return grad_f.grad_f_mem


def get_idx_set(grad_f):
    return grad_f.idx_set


# TODO: double-check the else branch
@profile
def grad_energy(grad_f, C, scf_style=False):
    np.matmul(grad_f.f.F, C[:, grad_f.idx_set], out=grad_f.grad_f_mem)
    np.multiply(grad_f.grad_f_mem, 2 * (1 if scf_style else 2), out=grad_f.grad_f_mem)
    return grad_f.grad_f_mem


# def grad_energy_amor(grad_f, pert_ind, t): #swept version
#     grad_f.grad_f_mem[:, pert_ind] += 2 * ((cos(t) - 1) * grad_f.f.fc + grad_f.f.fs * sin(t))
#     return grad_f.grad_f_mem


# only retracts the columns in idx_set (and thus assumes those columns are orthogonal to the rest of C)
# Requires DeltaC's columns to match the columns of C in idx_set, and that DeltaC is orthogonal to C in idx_set
# sync = False means that the out array agrees with C in the frozen columns already.
@profile
def retract(C, DeltaC, lam, idx_set=slice(None), out=None, sync=True, fix_sign=False):
    C_new = C[:, idx_set] if lam == 0 else C[:, idx_set] + lam * DeltaC
    result, R = np.linalg.qr(C_new)

    if fix_sign:
        result[:, np.where(np.diag(R) < 0)] *= -1

    if out is None:
        out = C.copy()
    elif sync and out is not C:
        np.copyto(out, C)
    out[:, idx_set] = result
    return out


# NOTE: not the symmetric projection, assumes X is like FC where F is symmetric
# If slicing, X needs to be fed in while sliced.
def proj(C, X):
    return X - C @ (C.T @ X)
    # Y = C.T @ X[:, idx_set]
    # return X[:, idx_set] - C @ (Y + Y.T) / 2


# test/check for correctness and then use--no need to store fc/fs
# assume pert_ind is in idx_set


# stale should be True if you don't need information on the (projected) gradient outside of the idx_set columns; otherwise it should be False
# (assumes scf_style gradient)
@profile
def mi_amor(f, grad_f, C0, pert_vec, moved_e, f_ref, fast_proj=False, pgf=None):
    idx_set = grad_f.idx_set
    pert_ind = (
        moved_e
        if isinstance(idx_set, slice) and idx_set == slice(None)
        else grad_f.g2l[moved_e]
    )

    # fc = grad_f.grad_f_mem[:, pert_ind]/2
    fc2 = grad_f.grad_f_mem[:, pert_ind]
    fs = f.F @ pert_vec
    old_vec = C0[:, moved_e]
    t1 = dot(old_vec, fc2) / 2
    t2 = dot(pert_vec, fs)
    t3 = dot(pert_vec, fc2) / 2

    tmin = atan(2 * t3 / (t1 - t2)) / 2
    qty = 2 * t3 * sin(2 * tmin) + (t1 - t2) * cos(
        2 * tmin
    )  # proportional to negative of the second deriv
    min_t, next_eval = (
        (tmin, f_ref + ((t2 - t1 + qty) / 2))
        if qty < 0
        else (tmin + np.pi / 2, f_ref + ((t2 - t1 - qty) / 2))
    )
    ct, st = cos(min_t), sin(min_t)
    feps = (ct - 1) * fc2 + (2 * st) * fs
    grf_pert_ind = feps + fc2
    grad_f.grad_f_mem[:, pert_ind] = grf_pert_ind

    if not (fast_proj):
        C0[:, moved_e] = ct * old_vec + st * pert_vec
        return min_t, C0, next_eval, grad_f.grad_f_mem
    eps = (ct - 1) * old_vec + st * pert_vec
    nc = old_vec + eps

    gradf = pgf.grad_f.grad_f_mem  # 2FC^{m+1}, already updated at col pert_ind
    Ct_feps = C0.T @ feps
    pgf.proj_grad[:, pert_ind] += feps - C0 @ (Ct_feps)

    Ct_feps[moved_e] = dot(feps, nc)  # now is eps.T @ gradf

    pgf.proj_grad = dger(
        -1.0, old_vec, Ct_feps[idx_set], a=pgf.proj_grad, overwrite_a=1
    )
    pgf.proj_grad = dger(-1.0, eps, (nc).T @ gradf, a=pgf.proj_grad, overwrite_a=1)

    C0[:, moved_e] = nc
    return min_t, C0, next_eval, pgf.proj_grad


# NEEDS TO BE VERIFIED -- should go to the critical point nearest the current point on hte geodesic, be it a maximum or a minimum.
@profile
def mi_amor_es(f, grad_f, C0, pert_vec, moved_e, f_ref, fast_proj=False, pgf=None):
    idx_set = grad_f.idx_set
    pert_ind = (
        moved_e
        if isinstance(idx_set, slice) and idx_set == slice(None)
        else grad_f.g2l[moved_e]
    )

    # fc = grad_f.grad_f_mem[:, pert_ind]/2
    fc2 = grad_f.grad_f_mem[:, pert_ind]
    fs = f.F @ pert_vec
    old_vec = C0[:, moved_e]
    t1 = dot(old_vec, fc2) / 2
    t2 = dot(pert_vec, fs)
    t3 = dot(pert_vec, fc2) / 2

    tmin = atan(2 * t3 / (t1 - t2)) / 2
    qty = 2 * t3 * sin(2 * tmin) + (t1 - t2) * cos(2 * tmin)

    # i.e. to the nearest stationary point
    min_t, next_eval = tmin, f_ref + ((t2 - t1 + qty) / 2)

    ct, st = cos(min_t), sin(min_t)
    feps = (ct - 1) * fc2 + (2 * st) * fs
    grf_pert_ind = feps + fc2
    grad_f.grad_f_mem[:, pert_ind] = grf_pert_ind

    if not (fast_proj):
        C0[:, moved_e] = ct * old_vec + st * pert_vec
        return C0, next_eval, grad_f.grad_f_mem
    eps = (ct - 1) * old_vec + st * pert_vec
    nc = old_vec + eps

    gradf = pgf.grad_f.grad_f_mem  # 2FC^{m+1}, already updated at col pert_ind
    Ct_feps = C0.T @ feps
    pgf.proj_grad[:, pert_ind] += feps - C0 @ (Ct_feps)

    Ct_feps[moved_e] = dot(feps, nc)  # now is eps.T @ gradf

    pgf.proj_grad = dger(
        -1.0, old_vec, Ct_feps[idx_set], a=pgf.proj_grad, overwrite_a=1
    )
    pgf.proj_grad = dger(-1.0, eps, (nc).T @ gradf, a=pgf.proj_grad, overwrite_a=1)

    C0[:, moved_e] = nc
    return (C0, next_eval, pgf.proj_grad)
