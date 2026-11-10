import numpy as np
import time
from oracles import *
from line_profiler import profile


@profile
def cg_inner_loop(
    u_tensor,
    f_prev,
    f,
    grad_f,
    eps=1e-8,
    expressive=False,
    opt_profiling=False,
    ITER_MAX=10000,
    idx_set=slice(None),
):
    iter = 0
    n_elem = u_tensor.size

    normsq = lambda x: np.dot(x.ravel(), x.ravel())
    if expressive:
        print("Starting CG subroutine")
    del_E = f_prev

    res_prev = 100
    p_prev = np.zeros_like(get_curr_grad(grad_f))
    u_buffer = np.zeros(np.shape(u_tensor))
    np.copyto(u_buffer, u_tensor)

    lambda_prev = 1
    f_curr = f_prev
    if opt_profiling:
        sub_energies = [f_prev]

    while np.abs(del_E) > eps:

        if iter > ITER_MAX:
            print("CG for SCF failed to converge within {} iterations".format(ITER_MAX))
            return

        grad_curr = grad_energy(grad_f, u_tensor, scf_style=True)

        grad_curr = proj(u_tensor, grad_curr)
        res_curr = normsq(grad_curr)

        beta = 0 if iter % n_elem == 0 else res_curr / res_prev

        p_curr = grad_curr

        if beta != 0:
            p_curr += beta * p_prev

        def try_scale(scale):
            nonlocal u_buffer
            if scale == 0:
                return f_curr

            u_buffer = make_step(
                u_buffer, u_tensor, grad_curr, scale * lambda_prev, idx_set
            )
            a = energy(f, u_buffer, scf_style=True)
            return a

        scale, f_tmp = quadratic_fit_1d(try_scale, f_curr)

        lambda_curr = scale * lambda_prev
        if lambda_curr > 0 and f_tmp < f_prev:
            np.copyto(u_tensor, u_buffer)
            f_curr = f_tmp
        else:  # GD fall back
            # println("Fallback: line search: $(scale)")
            if beta != 0:
                p_curr += -beta * p_prev

            f_curr, lambda_curr, _ = line_search(
                u_buffer,
                u_tensor,
                f_curr,
                grad_curr,
                res_curr,
                2 * lambda_prev,
                idx_set,
                f,
            )
            np.copyto(u_tensor, u_buffer)

        del_E = f_curr - f_prev

        if expressive:
            print(f"Iter {iter}: E = {f_curr} (ΔE = {del_E})")

        lambda_prev = lambda_curr
        f_prev = f_curr
        if opt_profiling:
            sub_energies.append(f_prev)
        np.copyto(p_prev, p_curr)
        res_prev = res_curr
        iter += 1

    return u_tensor, sub_energies if opt_profiling else None


@profile
def quadratic_fit_1d(g, v0):
    v1 = g(1)
    v2 = g(2)
    a = 0.5 * v0 - v1 + 0.5 * v2
    negative_2ab = -1.5 * v0 + 2 * v1 - 0.5 * v2
    b = -negative_2ab / (2 * a)
    v_b = g(b)
    return b, v_b


@profile
def make_step(u_buffer, u_tensor, p_curr, lam, idx_set):
    if lam == 0:
        return u_tensor
    return retract(
        u_tensor, p_curr, -1 * lam, idx_set, out=u_buffer, sync=False, fix_sign=True
    )


@profile
def line_search(u_buffer, u_tensor, f_curr, grad_curr, res_curr, step, idx_set, f):
    while True:
        Q = f_curr - 0.5 * step * res_curr
        u_buffer = make_step(u_buffer, u_tensor, grad_curr, step, idx_set)
        f_v = energy(f, u_buffer, scf_style=True)
        if (f_v - Q) <= 1e-12:
            return f_v, step, res_curr

        if step < 1e-14:
            u_buffer = make_step(u_buffer, u_tensor, grad_curr, -1 * step, idx_set)
            f_opp = energy(f, u_buffer, scf_style=True)
            print("f_curr - f_v: ", f_curr - f_v)
            print("f_curr - f_off: ", f_curr - f_opp)
            print("res_curr: ", res_curr)
            print("delta u: ", np.linalg.norm(u_buffer - u_tensor))
            raise Exception("STUCK!")
        step /= 2


@profile
def cg_scf(
    f,
    grad_f,
    C0,
    timing=False,
    eps=1e-6,
    expressive=False,
    ITER_MAX=10000,
    damp_param=0.4,
    opt_profiling=False,
    idx_set=slice(None),
):

    if timing:
        start_time = time.perf_counter()
        e_time = 0

    if not (isinstance(idx_set, slice) and idx_set == slice(None)):
        grad_f = localize_grad(grad_f, idx_set)

    idx_set = get_idx_set(grad_f)

    iter_scf = 0
    f_prev_scf = 0

    if timing:
        foo = time.perf_counter()
        f_curr_scf = energy(f, C0)
        e_time += time.perf_counter() - foo
    else:
        f_curr_scf = energy(f, C0)

    del_E_scf = f_curr_scf - f_prev_scf

    if opt_profiling:
        scf_profile = [f_curr_scf]
        sub_iter_energies = []
    u_tensor = np.copy(C0)

    if expressive:
        print("SCF iter {}: E = {} (ΔE = {})".format(iter_scf, f_curr_scf, del_E_scf))

    while abs(del_E_scf) > eps:
        if iter_scf > ITER_MAX:
            print("SCF failed to converge in {} iterations.".format(ITER_MAX))
            return

        f_curr_cg = energy(f, u_tensor, scf_style=True)
        if expressive:
            print("Energy for CG: {}".format(f_curr_cg))
        u_tensor, prof = cg_inner_loop(
            u_tensor,
            f_curr_cg,
            f,
            grad_f,
            eps=1e-8,
            expressive=expressive,
            opt_profiling=opt_profiling,
            ITER_MAX=ITER_MAX,
            idx_set=idx_set,
        )

        iter_scf += 1

        f_prev_scf = f_curr_scf

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
