import numpy as np
from scipy.linalg import qr
from oracles import *
from numpy import cos, sin, real, imag, dot, pi
import time
import math
from line_profiler import profile

# TODO: interpolative decomposition?


@profile
def normalize(v):
    norm = math.sqrt(np.dot(v, v))
    return norm, v / norm


@profile
def orth(G_in, C, c_value, n_eff, fast_proj=False, pqr=True):
    G = G_in if fast_proj else proj(C, G_in)
    k = int(np.floor(n_eff / c_value))  # should precompute
    if k <= 1:
        norms = np.einsum("ij,ij->j", G, G)
        mover = np.argmax(norms)
        nm = np.sqrt(norms[mover])
        return nm, G[:, mover] / nm, mover, []
    if pqr:
        _, p_idx = qr(G, mode="r", pivoting=True)  # replace with dgeqp3rk ??

    else:
        p_idx = np.argsort(np.einsum("ij,ij->j", G, G))[::-1]
    p_idx = p_idx.tolist()
    mover = p_idx[0]
    norm, ell = normalize(G[:, mover])
    return norm, ell, mover, p_idx[1:k]


@profile
def geodesic_curve(C, gr, col_order, c_value, n_eff, fast_proj=False, pqr=True):
    mover = 0
    if len(col_order) <= 0:  # use c to pick a new column
        if c_value == 0:
            col_order = np.random.permutation(n_eff).tolist()
            mover = col_order.pop(0)
            norm, col_k = (
                normalize(gr[:, mover])
                if fast_proj
                else normalize(gr[:, mover] - C @ (C.T @ gr[:, mover]))
            )
        else:
            norm, col_k, mover, col_order = orth(gr, C, c_value, n_eff, fast_proj, pqr)
    else:
        mover = col_order.pop(0)
        norm, col_k = (
            normalize(gr[:, mover])
            if fast_proj
            else normalize(gr[:, mover] - C @ (C.T @ gr[:, mover]))
        )

    return norm, col_order, mover, col_k


@profile
def cbcd_scf(
    f,
    grad_f,
    C0,
    c_value=1,
    timing=False,
    eps=1e-6,
    ITER_MAX=10000,
    expressive=False,
    damp_param=0.4,
    opt_profiling=False,
    fast_proj=True,
    pqr=True,
    idx_set=slice(None),
):
    if timing:
        start_time = time.perf_counter()
        e_time = 0
    u_tensor = np.asfortranarray(C0)

    grad_f = localize_grad(grad_f, idx_set)

    n_eff = get_curr_grad(grad_f).shape[1]
    pgf = make_proj_grad_energy(grad_f) if fast_proj else None

    idx_set = get_idx_set(grad_f)

    f_prev_scf = 0
    if timing:
        foo = time.perf_counter()
        f_curr_scf = energy(f, u_tensor)
        e_time += time.perf_counter() - foo
    else:
        f_curr_scf = energy(f, u_tensor)

    if opt_profiling:
        scf_profile = [f_curr_scf]
        sub_iter_energies = []
    del_E_scf = f_curr_scf - f_prev_scf
    iter_scf = 0

    while abs(del_E_scf) > eps:
        if iter_scf > ITER_MAX:
            print("SCF failed to converge in {} iterations.".format(ITER_MAX))
            return

        f_ref = energy(f, u_tensor, scf_style=True)
        if expressive:
            print("Energy for CG: {}".format(f_ref))
        u_tensor, prof = circle_optimize(
            f,
            grad_f,
            f_ref,
            u_tensor,
            c_value,
            n_eff,
            pgf=pgf,
            opt_profiling=opt_profiling,
            fast_proj=fast_proj,
            pqr=pqr,
            idx_set=idx_set,
        )
        iter_scf += 1
        f_prev_scf = f_curr_scf

        u_tensor = retract(u_tensor, u_tensor, 0, idx_set=idx_set, out=u_tensor)

        if timing:
            foo = time.perf_counter()
            f_curr_scf = energy(f, u_tensor, full_and_mix=(True, damp_param))
            e_time += time.perf_counter() - foo
        else:
            f_curr_scf = energy(f, u_tensor, full_and_mix=(True, damp_param))

        del_E_scf = f_curr_scf - f_prev_scf
        if expressive:
            print(
                "SCF iter {}: E = {} (ΔE = {})".format(iter_scf, f_curr_scf, del_E_scf)
            )
        if opt_profiling:
            scf_profile.append(f_curr_scf)
            sub_iter_energies.append(prof)

    if timing and opt_profiling:
        return (
            u_tensor,
            f_curr_scf,
            (time.perf_counter() - start_time, e_time),
            (scf_profile, sub_iter_energies),
        )
    elif timing:
        return u_tensor, f_curr_scf, (time.perf_counter() - start_time, e_time)
    elif opt_profiling:
        return u_tensor, f_curr_scf, (scf_profile, sub_iter_energies)

    return u_tensor, f_curr_scf


@profile
def circle_optimize(
    f,
    grad_f,
    f_ref,
    C0,
    c_value,
    n_eff,
    eps=1e-8,
    ITER_MAX=100000,
    pgf=None,
    opt_profiling=False,
    fast_proj=True,
    pqr=True,
    idx_set=slice(None),
):
    difference = 100
    iter = 0
    gr = grad_energy(grad_f, C0, scf_style=True)
    if fast_proj:
        gr = proj_grad(pgf, C0)
    cons_diffs = 0

    col_order = []

    thr = 2 * n_eff + 1 if c_value == 0 else (n_eff // c_value) + 1
    conv_num = eps / n_eff

    min_t = np.pi / 2

    if opt_profiling:
        sub_energies = [f_ref]

    slicing = not (isinstance(idx_set, slice) and idx_set == slice(None))
    max_iter = ITER_MAX * n_eff

    while (
        ((np.abs(min_t) > 1e-5) and np.abs(min_t - np.pi) > 1e-5)
        if c_value == n_eff
        else (cons_diffs < thr and iter < max_iter)
    ):

        if difference < conv_num:  # FUTURE WORK: as a restarting heuristic?
            cons_diffs = cons_diffs + 1
        else:
            cons_diffs = 0

        iter += 1

        norm, col_order, mover, col_k = geodesic_curve(
            C0, gr, col_order, c_value, n_eff, fast_proj, pqr
        )

        if norm < 1e-10:
            difference = 0
            next_eval = f_ref
            if opt_profiling:
                sub_energies.append(f_ref)
            continue

        mover = idx_set[mover] if slicing else mover

        min_t, C0, next_eval, gr = mi_amor(
            f, grad_f, C0, col_k, mover, f_ref, fast_proj, pgf
        )

        # min_t, next_eval = energy_amor(f, grad_f, C0, col_k, mover, f_ref)
        # gr = grad_energy_amor(grad_f, mover, min_t) # using only cheap gradient *updates*
        # if fast_proj: gr = proj_grad_amor(pgf, C0, col_k, mover, min_t)

        # if min_t > pi/2:
        #     min_t -= pi

        # C0[:, mover] = C0[:, mover] * np.cos(min_t) + col_k * np.sin(min_t)

        difference = abs(next_eval - f_ref)

        # if np.mod(iter - last_retr, ret_num) == ret_num - 1 or cons_diffs >= check: # once-in-a-while retractions; FUTURE WORK: tuning this knob
        #     # Note that this step is cubic (the scf_style updates are juist doing Tr[C^T FC])
        #     C0 = retract(C0, C0, 0)

        #     next_eval = energy(f, C0, scf_style=True)
        #     gr = grad_energy(grad_f, C0, scf_style=True)
        #     if fast_proj: gr = proj_grad(pgf, C0)
        #     difference = abs(next_eval - f_ref)
        #     last_retr = iter

        f_ref = next_eval

        if opt_profiling:
            sub_energies.append(f_ref)

    return C0, sub_energies if opt_profiling else None
