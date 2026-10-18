"""
Idea 2 dataset: high-frequency Helmholtz  -Lap u - (omega/c(x))^2 u = f  on the unit square,
surrounded by a PML; zero Dirichlet on the outer edge of the PML.

Discretisation (mirrored exactly by julia/MMDData/src/helmholtz.jl):
  node grid, spacing h = 1/N on the physical square, plus `npml` nodes on each side;
  node i (1-based) sits at x_i = (i - npml - 0.5) h; physical region is 0 < x < 1.
  Complex stretching s(x) = 1 + i sigma(x)/omega, sigma = sigma0 (d/Lp)^2 inside the PML,
  d = distance into the PML, Lp = npml h, sigma0 = 3 ln(1e6) c_max / (2 Lp)
  (time convention exp(-i omega t)).
  Row (i,j):  sum_faces a_f (u_c - u_nb) - (1 + i*beta) (omega/c)^2 sx(x_i) sy(y_j) u_c = sx sy f
  with x-face coefficient a = sy(y_j) / sx(x_{i +- 1/2}) / h^2, y-face analogously.
  beta = 0 gives the Helmholtz matrix; beta = +0.5 gives the usual shifted-Laplacian
  preconditioner (1 + 0.5i) k^2 (sign must match the damping of exp(-i omega t)).

Frequency sweep: freq = number of wavelengths across the domain at c = c_min = 1,
omega = 2 pi freq; the grid keeps a fixed 10 points per (shortest) wavelength, so
N = ceil(ppw * freq). This is the standard sweep under which classical
preconditioners degrade.

Media (defined in continuous coordinates, sampled per grid, extended into the PML by
clamping to the nearest physical point; all scaled so min c = 1):
  constant    c = 1
  layered     4-8 gently curved layers, c in [1, 3]
  smooth      c = 1 + 1.5 * sigmoid(GRF), correlation length 0.1
  inclusions  background c = 2, 8-25 slow circular scatterers with c = 1

Output: one HDF5 per (medium, split) with groups /freq_<F>/ holding
  c (n, nt, nt)  sound speed on the full grid (incl. PML), nt = N + 2 npml
  f (n, nt, nt)  real right-hand side (narrow Gaussian source, random location)
  attrs: freq, omega, N, npml, h, ppw, sigma0
and /check/ reference (x, y = A x) at the lowest frequency for the Julia self-check.
"""
from __future__ import annotations

import hashlib
import os

import h5py
import numpy as np
import scipy.sparse as sp

from diffusion import grf

PPW = 10
FREQS = [2, 4, 8, 16, 32]


def _seed(*parts):
    return int.from_bytes(hashlib.sha1("|".join(map(str, parts)).encode()).digest()[:4], "little")


def grid(N, npml):
    h = 1.0 / N
    nt = N + 2 * npml
    x = (np.arange(1, nt + 1) - npml - 0.5) * h
    return h, nt, x


def sigma(x, Lp, sigma0):
    d = np.maximum(0.0, np.maximum(-x, x - 1.0))
    return sigma0 * (d / Lp) ** 2


def assemble(c, freq, N, npml, beta=0.0):
    """Complex sparse matrix; unknown (i,j) -> i + nt*j (Julia column-major)."""
    h, nt, x = grid(N, npml)
    omega = 2 * np.pi * freq
    Lp = npml * h
    sigma0 = 3 * np.log(1e6) * c.max() / (2 * Lp)
    s = lambda xx: 1 + 1j * sigma(xx, Lp, sigma0) / omega
    sn = s(x)                    # at nodes
    sf = s(x + 0.5 * h)          # at face i+1/2, index i (face between node i and i+1)
    SX, SY = np.meshgrid(sn, sn, indexing="ij")
    # x-faces: between (i,j) and (i+1,j): sy(y_j)/sx(x_{i+1/2})/h^2
    ax = (sn[None, :] / sf[:-1, None]) / h ** 2       # shape (nt-1, nt): face i (between i,i+1), row j
    ay = (sn[:, None] / sf[None, :-1]) / h ** 2       # shape (nt, nt-1)
    # faces to the outer Dirichlet wall use the ghost-face stretching s(x_1 - h/2) / s(x_nt + h/2)
    s_left = s(x[0] - 0.5 * h); s_right = sf[-1]
    aw = np.empty((nt, nt), complex); ae = np.empty((nt, nt), complex)
    as_ = np.empty((nt, nt), complex); an = np.empty((nt, nt), complex)
    aw[0, :] = sn / s_left / h ** 2; aw[1:, :] = ax
    ae[-1, :] = sn / s_right / h ** 2; ae[:-1, :] = ax
    as_[:, 0] = sn / s_left / h ** 2; as_[:, 1:] = ay
    an[:, -1] = sn / s_right / h ** 2; an[:, :-1] = ay
    diag = aw + ae + as_ + an - (1 + 1j * beta) * (omega / c) ** 2 * SX * SY
    idx = np.arange(nt * nt).reshape((nt, nt), order="F")
    I, J = np.meshgrid(np.arange(nt), np.arange(nt), indexing="ij")
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
    for a, di, dj in ((aw, -1, 0), (ae, 1, 0), (as_, 0, -1), (an, 0, 1)):
        ok = (I + di >= 0) & (I + di < nt) & (J + dj >= 0) & (J + dj < nt)
        rows.append(idx[ok]); cols.append(idx[I[ok] + di, J[ok] + dj]); vals.append(-a[ok])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nt * nt, nt * nt))
    return A, SX * SY, sigma0


def rhs(f, N, npml, c, freq):
    _, sxsy, _ = assemble_scaling(N, npml, c, freq)
    return (sxsy * f).ravel(order="F")


def assemble_scaling(N, npml, c, freq):
    h, nt, x = grid(N, npml)
    omega = 2 * np.pi * freq
    Lp = npml * h
    sigma0 = 3 * np.log(1e6) * c.max() / (2 * Lp)
    sn = 1 + 1j * sigma(x, Lp, sigma0) / omega
    SX, SY = np.meshgrid(sn, sn, indexing="ij")
    return h, SX * SY, sigma0


# ------------------------------------------------------------------ media
def medium(kind, rng):
    if kind == "constant":
        return lambda X, Y: np.ones_like(X)
    if kind == "layered":
        nl = rng.integers(4, 9)
        depths = np.sort(rng.uniform(0.05, 0.95, nl - 1))
        amps = rng.uniform(0, 0.05, nl - 1); fr = rng.uniform(0.5, 2, nl - 1); ph = rng.uniform(0, 6.28, nl - 1)
        vals = rng.uniform(1, 3, nl); vals = vals / vals.min()
        def c(X, Y):
            layer = np.zeros(X.shape, int)
            for d, a, f_, p in zip(depths, amps, fr, ph):
                layer += Y > d + a * np.sin(2 * np.pi * f_ * X + p)
            return vals[layer]
        return c
    if kind == "smooth":
        sd = int(rng.integers(2 ** 31))
        def c(X, Y):
            n = X.shape[0]
            phi = grf(n, np.random.default_rng(sd), ell=0.1, kind="gaussian")
            v = 1 + 1.5 / (1 + np.exp(-1.5 * phi))
            return v / v.min()
        c.grid_based = True  # GRF is sampled on the physical grid, then edge-padded into the PML
        return c
    if kind == "inclusions":
        nc = rng.integers(8, 26)
        cx, cy = rng.uniform(0.1, 0.9, nc), rng.uniform(0.1, 0.9, nc)
        r = rng.uniform(0.02, 0.06, nc)
        def c(X, Y):
            v = np.full(X.shape, 2.0)
            for a, b, rr in zip(cx, cy, r):
                v[(X - a) ** 2 + (Y - b) ** 2 < rr ** 2] = 1.0
            return v
        return c
    raise ValueError(kind)


def sample(cfun, N, npml):
    h, nt, x = grid(N, npml)
    xc = np.clip(x, 0.5 * h, 1 - 0.5 * h)       # clamp into the physical square
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    if getattr(cfun, "grid_based", False):
        Xp, Yp = np.meshgrid((np.arange(N) + 0.5) * h, (np.arange(N) + 0.5) * h, indexing="ij")
        return np.pad(cfun(Xp, Yp), npml, mode="edge")
    return cfun(X, Y)


def source(N, npml, rng):
    h, nt, x = grid(N, npml)
    X, Y = np.meshgrid(x, x, indexing="ij")
    x0, y0 = rng.uniform(0.2, 0.8, 2)
    w = 2 * h
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * w ** 2)) / (2 * np.pi * w ** 2)


def write_split(out, kind, split, n_inst, freqs=FREQS):
    os.makedirs(out, exist_ok=True)
    base = 0 if split == "train" else 10_000
    path = os.path.join(out, f"helmholtz_{kind}_{split}.h5")
    seeds = [_seed(kind, base + i) for i in range(n_inst)]
    with h5py.File(path, "w") as hf:
        hf.attrs.update(medium=kind, split=split, ppw=PPW, seeds=seeds,
                        convention="exp(-i omega t); see python/helmholtz.py docstring")
        for F in freqs:
            N = int(np.ceil(PPW * F))
            npml = max(10, PPW)
            C, S = [], []
            for s in seeds:
                rng = np.random.default_rng(s)
                cfun = medium(kind, rng)
                C.append(sample(cfun, N, npml))
                S.append(source(N, npml, np.random.default_rng(s + F)))
            g = hf.create_group(f"freq_{F}")
            g.attrs.update(freq=F, omega=2 * np.pi * F, N=N, npml=npml, h=1.0 / N)
            g.create_dataset("c", data=np.stack(C), compression="gzip")
            g.create_dataset("f", data=np.stack(S), compression="gzip")
        # self-check at the lowest frequency
        F = freqs[0]; N = int(np.ceil(PPW * F)); npml = max(10, PPW)
        c0 = hf[f"freq_{F}/c"][0]
        A, sxsy, s0 = assemble(c0, F, N, npml)
        nt = c0.shape[0]
        xr = np.random.default_rng(7).standard_normal((nt, nt))
        y = A @ xr.ravel(order="F")
        As, _, _ = assemble(c0, F, N, npml, beta=0.5)
        ys = As @ xr.ravel(order="F")
        g = hf.create_group("check")
        g.attrs.update(freq=F, N=N, npml=npml, sigma0=s0, what="instance 0 at lowest freq; y = A*vec(x); y_shifted uses beta=+0.5")
        g.create_dataset("x", data=xr)
        g.create_dataset("y_re", data=y.real.reshape((nt, nt), order="F"))
        g.create_dataset("y_im", data=y.imag.reshape((nt, nt), order="F"))
        g.create_dataset("ys_re", data=ys.real.reshape((nt, nt), order="F"))
        g.create_dataset("ys_im", data=ys.imag.reshape((nt, nt), order="F"))
    return path


def build(out, n_train=16, n_test=8):
    for kind in ("constant", "layered", "smooth", "inclusions"):
        for split, n in (("train", n_train), ("test", n_test)):
            print("wrote", write_split(out, kind, split, n))


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "helmholtz")
    build(out)
