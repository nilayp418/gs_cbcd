import os
import numpy as np
import scipy as sp
from pyscf import gto, scf
from oracles import (
    build_integrals,
    make_rhf_oracle_energy,
    make_rhf_grad_energy,
    energy,
    proj,
)
from python_impls.cg_scf import cg_scf
from python_impls.cbcd_scf import cbcd_scf
from python_impls.pure_scf import pure_scf

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref_data")


def load_molecule(molecule, basis):
    with open(os.path.join(REF, f"{molecule}.txt")) as fh:
        mol_atom_str = "".join(fh.readlines()[1:])

    mol = gto.Mole()
    mol.atom = mol_atom_str
    mol.basis = basis
    mol.build()
    ne = mol.nelectron
    N_o = ne // 2
    n_effs = list(range(1, N_o + 1))
    integ = build_integrals(mol)
    f = make_rhf_oracle_energy(integ, ne)

    mf_guess = scf.RHF(mol)
    mf_guess.max_cycle = 0
    mf_guess.kernel()
    S = mf_guess.get_ovlp()
    X = np.real(sp.linalg.fractional_matrix_power(S, 0.5))
    C0_orth = X @ mf_guess.mo_coeff[:, : ne // 2]

    mf = scf.RHF(mol)
    e_pyscf = mf.kernel() - mol.get_enuc()

    return mol, integ, f, ne, N_o, n_effs, C0_orth, e_pyscf


def find_reference_solution(f, C0_orth, dp, etar, eps=1e-6, ITER_MAX=10000):
    mat_ref, e_guess = pure_scf(
        f,
        np.copy(C0_orth),
        timing=False,
        eps=eps,
        ITER_MAX=ITER_MAX,
        expressive=False,
        damp_param=dp,
    )
    if abs(e_guess - etar) > 1e-4:
        print(
            f"ERROR: Energy of guess not close PySCF energy. Variational Error: {e_guess - etar}"
        )
        return
    orbital_energies = np.diag(mat_ref.T @ f.F @ mat_ref)
    return mat_ref, orbital_energies


def make_impurity_guess(mat_ref, C0_orth, n_eff, rng=None, reproduce_error=False):

    if reproduce_error:
        # Fixed seed=8 reliably reproduces a live bug for SO2/aug-cc-pVTZ,
        # n_eff=1: this particular random Gaussian draw sends Local CG's inner
        # loop to ITER_MAX without converging, and cg_inner_loop's None return
        # then crashes cg_scf's unpacking.
        rng = np.random.default_rng(8)

    mat = (
        rng.normal(size=C0_orth.shape)
        if isinstance(rng, np.random.Generator)
        else C0_orth
    )
    pivoting = not (isinstance(rng, np.random.Generator))

    # pivoting=False returns a 2-tuple (Q, R); pivoting=True returns 3 (Q, R, piv) --
    # Q is always first, so unpack the rest loosely to handle both.
    Q, *_ = sp.linalg.qr(
        (np.eye(mat_ref.shape[0]) - mat_ref[:, :-n_eff] @ mat_ref[:, :-n_eff].T) @ mat,
        pivoting=pivoting,
    )
    C_orth_imp = np.copy(mat_ref)
    C_orth_imp[:, -n_eff:] = Q[:, :n_eff]
    idx_set = [C_orth_imp.shape[1] - i - 1 for i in range(n_eff)][::-1]
    return C_orth_imp, idx_set


def projected_gradient_norm(f, C):
    """||(I - CC^T) F C|| at the given point. Near zero means C is a genuine
    stationary point of the energy (e.g. an alternate, non-global solution),
    not just a run that failed to converge -- distinguishes the two when a
    chemical-accuracy check fails."""
    energy(f, C)  # refresh f.F to reflect C's density
    return 2 * np.linalg.norm(proj(C, f.F @ C))


def run_method(f, c_value, C_init, idx_set, dp, fast_proj=False, **kwargs):
    """Dispatch one solver run.
    c_value >= 0  : CBCD with that c (idx_set-restricted)
    c_value == -1 : Local CG (idx_set-restricted)
    c_value == -2 : Global CG (idx_set=slice(None))
    c_value == -3 : SCF (pure_scf, no idx_set)
    grad is rebuilt fresh on every call -- localize_grad is one-time-use, so it
    cannot be reused across separate solver invocations.
    Extra kwargs (timing=, opt_profiling=, expressive=, eps=, ITER_MAX=) pass
    straight through to the underlying solver.
    """
    if c_value == -3:
        return pure_scf(f, C_init, damp_param=dp, **kwargs)
    grad = make_rhf_grad_energy(f)
    if c_value == -2:
        return cg_scf(f, grad, C_init, damp_param=dp, **kwargs)
    if c_value == -1:
        return cg_scf(f, grad, C_init, damp_param=dp, idx_set=idx_set, **kwargs)
    return cbcd_scf(
        f,
        grad,
        C_init,
        c_value,
        damp_param=dp,
        idx_set=idx_set,
        fast_proj=fast_proj,
        **kwargs,
    )
