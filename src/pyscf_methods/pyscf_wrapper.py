#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyscf import gto, scf
import numpy as np
from oracles import *
import scipy as sp
from python_impls.pure_scf import *
from python_impls.cg_scf import *
from python_impls.cbcd_scf import *
import matplotlib.pyplot as plt
import awkward as ak
from pprint import pprint
import json
from datetime import datetime
import os
import re

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref_data")

# In[2]:


molecules = ["SO2"]
dp_list = [0.4 for i in range(len(molecules))]
timings = {}
adjusted_timings = {}


# In[3]:


for molecule, dp in zip(molecules, dp_list):
    print(f"Molecule: {molecule}")
    with open(os.path.join(REF, f"{molecule}.txt")) as f:
        mol_atom_str = "".join(f.readlines()[1:])  # skip header row

    mol = gto.Mole()
    # mol.output = 'out_h2o'
    mol.atom = mol_atom_str
    mol.basis = "aug-cc-pVTZ"
    mol.build()
    ne = mol.nelectron
    integ = build_integrals(mol)
    f = make_rhf_oracle_energy(integ, ne)
    grad = make_rhf_grad_energy(f)

    mf_guess = scf.RHF(mol)
    mf_guess.max_cycle = 0
    print(f"Initial non-nuclear_E : {mf_guess.kernel() - mol.get_enuc()}")

    S = mf_guess.get_ovlp()
    X = np.real(sp.linalg.fractional_matrix_power(S, 0.5))
    C0_orth = X @ mf_guess.mo_coeff[:, : ne // 2]

    mf = scf.RHF(mol)
    e_pyscf = mf.kernel() - mol.get_enuc()
    print(f"non-nuclear_E : {e_pyscf}")

    # print(np.linalg.norm(C0_orth.T @ C0_orth - np.eye(C0_orth.shape[1])))  # ≈ 0

    runs = 11
    timing_cg = []
    timing_cbcd1 = []
    timing_cbcd0 = []
    timing_cbcd2 = []
    timing_cbcd_ncols = []
    timing_scf = []

    e_times_cg = []
    e_times_cbcd1 = []
    e_times_cbcd2 = []
    e_times_cbcd_ncols = []
    e_times_cbcd0 = []
    e_times_scf = []

    print("SCF")
    for i in range(runs):
        C_init = np.copy(C0_orth)
        mat, etar, (tscf, et) = pure_scf(
            f,
            C_init,
            timing=True,
            eps=1e-6,
            ITER_MAX=10000,
            expressive=False,
            damp_param=dp,
        )
        timing_scf.append(tscf)
        e_times_scf.append(tscf - et)

    print(f"Variational Error? {(etar - e_pyscf)}")
    print(f"Median Time over {runs} runs: {np.median(timing_scf)}")
    print(f"Non-N^4 Time over {runs} runs: {np.median(e_times_scf)}")

    if etar - e_pyscf > 1e-4:
        print("ERROR: Energy not in agreement with PySCF to chemical accuracy")
        break

    f = make_rhf_oracle_energy(integ, ne)
    grad = make_rhf_grad_energy(f)

    print("CG")
    for i in range(runs):
        C_init = np.copy(C0_orth)
        mat, etar, (tcg, et) = cg_scf(
            f,
            grad,
            C_init,
            timing=True,
            eps=1e-6,
            ITER_MAX=10000,
            expressive=False,
            damp_param=dp,
        )
        timing_cg.append(tcg)
        e_times_cg.append(tcg - et)

    print(f"Variational Error? {(etar - e_pyscf)}")
    print(f"Median Time over {runs} runs: {np.median(timing_cg)}")
    print(f"Non-N^4 Time over {runs} runs: {np.median(e_times_cg)}")

    if etar - e_pyscf > 1e-4:
        print("ERROR: Energy not in agreement with PySCF to chemical accuracy")
        break

    f = make_rhf_oracle_energy(integ, ne)
    grad = make_rhf_grad_energy(f)

    # c_value = 1
    # print(f"CBCD, c = {c_value}")

    # for i in range(runs):
    #     C_init = np.copy(C0_orth)
    #     mat, etar, (tcbcd, et) = cbcd_scf(f, grad, C_init, c_value, timing=True, eps=1e-6, ITER_MAX=10000, expressive=False, damp_param=dp, fast_proj=False, pqr=False)
    #     timing_cbcd1.append(tcbcd)
    #     e_times_cbcd1.append(tcbcd-et)

    # print(f"Variational Error? {(etar - e_pyscf)}")
    # print(f"Median Time over {runs} runs: {np.median(timing_cbcd1)}")
    # print(f"Non-N^4 Time over {runs} runs: {np.median(e_times_cbcd1)}")

    # if etar - e_pyscf > 1e-4:
    #     print("ERROR: Energy not in agreement with PySCF to chemical accuracy")
    #     break

    c_value = mat.shape[1]
    print(f"CBCD, c = {c_value} (columns in C)")

    for i in range(runs):
        C_init = np.copy(C0_orth)
        mat, etar, (tcbcd, et) = cbcd_scf(
            f,
            grad,
            C_init,
            c_value,
            timing=True,
            eps=1e-6,
            ITER_MAX=10000,
            expressive=False,
            damp_param=dp,
            fast_proj=True,
        )
        timing_cbcd_ncols.append(tcbcd)
        e_times_cbcd_ncols.append(tcbcd - et)

    print(f"Variational Error? {(etar - e_pyscf)}")
    print(f"Median Time over {runs} runs: {np.median(timing_cbcd_ncols)}")
    print(f"Non-N^4 Time over {runs} runs: {np.median(e_times_cbcd_ncols)}")

    if etar - e_pyscf > 1e-4:
        print("ERROR: Energy not in agreement with PySCF to chemical accuracy")
        break
