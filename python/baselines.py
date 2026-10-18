"""
Baseline solver sweeps: confirms that the generated problem families actually break standard
solvers (the premise of Idea 2) and gives the reference curves new solvers must beat.

Diffusion: iterations of CG / Jacobi-PCG / ILU(0)-PCG / AMG-PCG (pyamg smoothed aggregation)
to ||r|| / ||b|| <= 1e-8 as the contrast grows on fixed geometry.
Helmholtz: iterations of GMRES and GMRES + shifted-Laplacian (exact inverse, the "ideal"
version of the standard preconditioner) as the frequency grows at 10 points per wavelength.

Writes results/baselines_*.csv and a figure.
"""
from __future__ import annotations

import csv
import os
import time

import h5py
import numpy as np
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import diffusion as D
import helmholtz as H

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
CAP = 5000


def pcg_iters(A, b, M=None, tol=1e-8, maxiter=CAP):
    it = [0]
    cb = lambda xk: it.__setitem__(0, it[0] + 1)
    _, info = sla.cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, M=M, callback=cb)
    return it[0] if info == 0 else np.nan


def diffusion_preconds(A):
    out = {"CG": None}
    d = A.diagonal()
    out["Jacobi-PCG"] = sla.LinearOperator(A.shape, matvec=lambda x: x / d)
    # symmetric Gauss-Seidel (SSOR, omega=1): SPD, so valid for CG (a nonsymmetric ILU is not)
    Lw = sp.tril(A, format="csr"); Up = sp.triu(A, format="csr")
    sgs = lambda r: sla.spsolve_triangular(Up, d * sla.spsolve_triangular(Lw, r, lower=True), lower=False)
    out["SGS-PCG"] = sla.LinearOperator(A.shape, matvec=sgs)
    ml = pyamg.smoothed_aggregation_solver(A.tocsr())
    out["AMG-PCG"] = ml.aspreconditioner(cycle="V")
    return out


def run_diffusion(N=128, n_inst=1):
    rows = []
    for fam in ("channels", "inclusions", "barriers"):
        h = h5py.File(os.path.join(DATA, "diffusion", f"diffusion_{fam}_N{N}_test.h5"))
        sign = int(h.attrs["sign"])
        for i in range(n_inst):
            mask, f = h["mask"][i], h["f"][i]
            b = D.rhs(f)
            for p in D.CONTRAST_EXPONENTS:
                A = D.assemble(D.k_binary(mask, p, sign))
                for name, M in diffusion_preconds(A).items():
                    rows.append(dict(family=fam, inst=i, knob="log10_contrast", value=p, solver=name,
                                     iters=pcg_iters(A, b, M)))
            print(fam, i, "done")
    for kind in ("exponential",):
        h = h5py.File(os.path.join(DATA, "diffusion", f"diffusion_lognormal_{kind}_N{N}_test.h5"))
        for i in range(n_inst):
            phi, f = h["phi"][i], h["f"][i]
            b = D.rhs(f)
            for s2 in h.attrs["sigma2"]:
                k = np.exp(np.sqrt(s2) * phi)
                A = D.assemble(k)
                for name, M in diffusion_preconds(A).items():
                    rows.append(dict(family=f"lognormal_{kind}", inst=i, knob="log10_contrast",
                                     value=float(np.log10(k.max() / k.min())), solver=name, iters=pcg_iters(A, b, M)))
            print("lognormal", i, "done")
    return rows


def gmres_iters(A, b, M=None, tol=1e-6, cap=1000, restart=200):
    """GMRES(restart) iterations (inner steps) to ||b - Ax|| / ||b|| <= tol, nan if > cap."""
    it = [0]
    cb = lambda r: it.__setitem__(0, it[0] + 1)
    x, info = sla.gmres(A, b, M=M, rtol=tol, atol=0.0, restart=restart, maxiter=max(1, cap // restart),
                        callback=cb, callback_type="pr_norm")
    ok = np.linalg.norm(b - A @ x) <= 1.01 * tol * np.linalg.norm(b)
    return it[0] if ok else np.nan


def run_helmholtz(n_inst=2):
    rows = []
    for kind in ("constant", "layered", "smooth", "inclusions"):
        h = h5py.File(os.path.join(DATA, "helmholtz", f"helmholtz_{kind}_test.h5"))
        for F in H.FREQS:
            g = h[f"freq_{F}"]
            N, npml = int(g.attrs["N"]), int(g.attrs["npml"])
            for i in range(n_inst):
                c, f = g["c"][i], g["f"][i]
                A, sxsy, _ = H.assemble(c, F, N, npml)
                b = (sxsy * f).ravel(order="F")
                As, _, _ = H.assemble(c, F, N, npml, beta=0.5)
                lu = sla.splu(As.tocsc())
                Msl = sla.LinearOperator(A.shape, matvec=lu.solve, dtype=complex)
                t = time.time()
                rows.append(dict(family=f"helmholtz_{kind}", inst=i, knob="freq", value=F, solver="GMRES",
                                 iters=gmres_iters(A, b, cap=1000)))
                rows.append(dict(family=f"helmholtz_{kind}", inst=i, knob="freq", value=F, solver="GMRES+SL(exact)",
                                 iters=gmres_iters(A, b, Msl, cap=1000)))
                print(kind, F, i, f"{time.time()-t:.1f}s", rows[-2]["iters"], rows[-1]["iters"])
    return rows


def save(rows, name):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["diffusion", "helmholtz"]
    if "diffusion" in which:
        save(run_diffusion(), "baselines_diffusion.csv")
    if "helmholtz" in which:
        save(run_helmholtz(), "baselines_helmholtz.csv")
