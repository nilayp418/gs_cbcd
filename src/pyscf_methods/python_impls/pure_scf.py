import numpy as np
import time
from oracles import energy
from line_profiler import profile


@profile
def pure_scf(
    f,
    C0,
    timing=False,
    eps=1e-6,
    ITER_MAX=10000,
    expressive=True,
    damp_param=0.4,
    opt_profiling=False,
):

    if timing:
        start_time = time.perf_counter()
        e_time = 0

    iter_scf = 0
    f_prev_scf = 0

    if timing:
        foo = time.perf_counter()
        f_curr_scf = energy(f, C0)
        e_time += time.perf_counter() - foo
    else:
        f_curr_scf = energy(f, C0)

    if opt_profiling:
        scf_profile = [f_curr_scf]
    del_E_scf = f_curr_scf - f_prev_scf

    while abs(del_E_scf) > eps:
        if iter_scf > ITER_MAX:
            print("SCF failed to converge in {} iterations.".format(ITER_MAX))
            return

        _, u_tensor = np.linalg.eigh(f.F)
        u_tensor = u_tensor[:, : C0.shape[1]]
        f_prev_scf = f_curr_scf

        if timing:
            foo = time.perf_counter()
            f_curr_scf = energy(f, u_tensor, full_and_mix=(True, damp_param))
            e_time += time.perf_counter() - foo
        else:
            f_curr_scf = energy(f, u_tensor, full_and_mix=(True, damp_param))

        del_E_scf = f_curr_scf - f_prev_scf
        iter_scf += 1
        if expressive:
            print(
                "SCF iter {}: E = {} (ΔE = {})".format(iter_scf, f_curr_scf, del_E_scf)
            )
        if opt_profiling:
            scf_profile.append(f_curr_scf)

    if timing and opt_profiling:
        return (
            u_tensor,
            f_curr_scf,
            (time.perf_counter() - start_time, e_time),
            scf_profile,
        )
    elif timing:
        return u_tensor, f_curr_scf, (time.perf_counter() - start_time, e_time)
    elif opt_profiling:
        return u_tensor, f_curr_scf, scf_profile

    return u_tensor, f_curr_scf
