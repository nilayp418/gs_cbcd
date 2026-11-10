This repository contains the code that implements the column-based coordinate descent procedure for ground-state electronic structure calculations, as well as notebooks demonstrating various use cases of these algorithms and the figures produced in this manuscript.
The required packages are listed in `requirements.txt.`. The code largely builds off of PySCF, and was developed in Python 3.14. For setup, run ```pip install -r requirements.txt```. 
For plotting, one should also install `pdflatex ` and for running the code in C one should also install ```clang```.

All calculations in this repository are done in the restricted Hartree-Fock (RHF) setting. The C code (written in code_in_c/) is compatible with macOS only. The routines are also implemented in Python.
Also included is data from the S66x8 dataset (https://doi.org/10.1039/d2cp03938a), as taken from Psi4, as well as equilibrium geometries of various molecules, as taken from the NIST dataset (https://cccbdb.nist.gov/).
The working directory is src/pyscf_methods, the S66x8 dataset information is in src/psi4, the geometries are in src/ref_data, and the figures are in manuscript/new_figures. 

## Citation

If you use this code, please cite:

N. Patel, K. Li, A. Damle, *Locally Adaptive Direct Minimization for Mean-Field Electronic Structure*, (submitted, 2026).
