"""
Idea 2 dataset: high-contrast diffusion  -div(k grad u) = f  on (0,1)^2, u = 0 on the boundary.

Discretisation (mirrored exactly by julia/MMDData/src/diffusion.jl):
  cell-centred finite volumes on an N x N grid, h = 1/N
  interior face transmissibility  2 k_a k_b / (k_a + k_b)
  boundary face transmissibility  2 k            (half-cell distance to the wall)
  A u = h^2 f   (the h^2 is moved to the right-hand side so A is O(k))

Coefficient families
  lognormal   k = exp(sigma * phi), phi a unit-variance Gaussian random field (ConDiff-style);
              contrast is controlled statistically by sigma^2
  channels    k = 1 + (10^p - 1) * mask, mask = meandering high-permeability channels
  inclusions  k = 1 + (10^p - 1) * mask, mask = random rectangular inclusions
  barriers    k = 1 + (10^-p - 1) * mask, mask = thin low-permeability barriers
For the binary families the *geometry* is stored once per instance (mask) and the contrast
p is a sweep parameter, so "iterations vs contrast" is measured on identical geometry.
Masks are defined in continuous coordinates, so N=64/128/256 resolve the same geometry.

Output: one HDF5 file per (family, N, split) with
  /k        (n, N, N)  float64   lognormal only
  /mask     (n, N, N)  uint8     binary families only
  /f        (n, N, N)  float64   right-hand side (unit-variance white noise, ConDiff-style)
  /check/*  reference x, y = A x for instance 0 (and contrast 10^4 for binary families),
            so the Julia assembly can verify itself
  attrs: family, N, split, contrasts (binary), sigma2 (lognormal), seeds
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import scipy.sparse as sp


# ------------------------------------------------------------------ assembly
def assemble(k: np.ndarray) -> sp.csr_matrix:
    """5-point FV matrix for k on an N x N grid; unknown (i,j) -> i + N*j (Julia column-major)."""
    N = k.shape[0]
    hm = lambda a, b: 2 * a * b / (a + b)
    tw = np.empty_like(k); te = np.empty_like(k); ts = np.empty_like(k); tn = np.empty_like(k)
    tw[0, :] = 2 * k[0, :];   tw[1:, :] = hm(k[:-1, :], k[1:, :])
    te[-1, :] = 2 * k[-1, :]; te[:-1, :] = hm(k[1:, :], k[:-1, :])
    ts[:, 0] = 2 * k[:, 0];   ts[:, 1:] = hm(k[:, :-1], k[:, 1:])
    tn[:, -1] = 2 * k[:, -1]; tn[:, :-1] = hm(k[:, 1:], k[:, :-1])
    idx = np.arange(N * N).reshape((N, N), order="F")
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [(tw + te + ts + tn).ravel()]
    for t, di, dj in ((tw, -1, 0), (te, 1, 0), (ts, 0, -1), (tn, 0, 1)):
        I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        ok = (I + di >= 0) & (I + di < N) & (J + dj >= 0) & (J + dj < N)
        rows.append(idx[ok]); cols.append(idx[I[ok] + di, J[ok] + dj]); vals.append(-t[ok])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N * N, N * N))
    return A


def rhs(f: np.ndarray) -> np.ndarray:
    N = f.shape[0]
    return (f / N ** 2).ravel(order="F")


# ------------------------------------------------------------------ fields
def centers(N):
    x = (np.arange(N) + 0.5) / N
    return np.meshgrid(x, x, indexing="ij")


def grf(N, rng, ell=0.05, kind="exponential"):
    """Unit-variance stationary GRF on the unit square, spectral method on a 2x periodic grid."""
    M = 2 * N
    kx = np.fft.fftfreq(M, d=1.0 / M) * 2 * np.pi  # angular wavenumber, domain length 2 -> scale below
    kx = kx / 2.0
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    if kind == "gaussian":
        S = np.exp(-0.5 * k2 * ell ** 2)
    elif kind == "exponential":  # Matern nu=1/2 in 2-D
        S = (1 + k2 * ell ** 2) ** (-1.5)
    else:
        raise ValueError(kind)
    w = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    phi = np.real(np.fft.ifft2(np.sqrt(S) * w))[:N, :N]
    return (phi - phi.mean()) / phi.std()


def geom_channels(rng):
    n = rng.integers(3, 9)
    ch = []
    for _ in range(n):
        ch.append(dict(y0=rng.uniform(0.05, 0.95), amp=rng.uniform(0.0, 0.08), freq=rng.uniform(0.5, 3),
                       ph=rng.uniform(0, 2 * np.pi), w=rng.uniform(0.012, 0.03),
                       x0=rng.uniform(0, 0.3), x1=rng.uniform(0.7, 1.0)))
    return ch


def mask_channels(N, ch):
    X, Y = centers(N)
    m = np.zeros((N, N), bool)
    for c in ch:
        yc = c["y0"] + c["amp"] * np.sin(2 * np.pi * c["freq"] * X + c["ph"])
        m |= (np.abs(Y - yc) < max(c["w"], 0.6 / N)) & (X >= c["x0"]) & (X <= c["x1"])
    return m


def geom_inclusions(rng):
    n = rng.integers(10, 41)
    return [dict(x=rng.uniform(0.05, 0.95), y=rng.uniform(0.05, 0.95),
                 a=rng.uniform(0.015, 0.06), b=rng.uniform(0.015, 0.06)) for _ in range(n)]


def mask_inclusions(N, inc):
    X, Y = centers(N)
    m = np.zeros((N, N), bool)
    for c in inc:
        m |= (np.abs(X - c["x"]) < max(c["a"], 0.6 / N)) & (np.abs(Y - c["y"]) < max(c["b"], 0.6 / N))
    return m


def geom_barriers(rng):
    n = rng.integers(4, 11)
    out = []
    for _ in range(n):
        th = rng.uniform(0, np.pi)
        L = rng.uniform(0.2, 0.6)
        out.append(dict(x=rng.uniform(0.15, 0.85), y=rng.uniform(0.15, 0.85), ux=np.cos(th), uy=np.sin(th),
                        L=L, w=rng.uniform(0.008, 0.02)))
    return out


def mask_barriers(N, bars):
    X, Y = centers(N)
    m = np.zeros((N, N), bool)
    for c in bars:
        dx, dy = X - c["x"], Y - c["y"]
        along = dx * c["ux"] + dy * c["uy"]
        perp = -dx * c["uy"] + dy * c["ux"]
        m |= (np.abs(along) < c["L"] / 2) & (np.abs(perp) < max(c["w"], 0.6 / N))
    return m


BINARY = {
    "channels": (geom_channels, mask_channels, +1),
    "inclusions": (geom_inclusions, mask_inclusions, +1),
    "barriers": (geom_barriers, mask_barriers, -1),
}
CONTRAST_EXPONENTS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
SIGMA2 = [0.1, 0.5, 1.0, 2.0, 4.0]


def k_binary(mask, p, sign):
    return 1.0 + (10.0 ** (sign * p) - 1.0) * mask.astype(float)


# ------------------------------------------------------------------ writer
def _seed(*parts):
    import hashlib
    return int.from_bytes(hashlib.sha1("|".join(map(str, parts)).encode()).digest()[:4], "little")


def write_split(out, family, N, split, n_inst):
    os.makedirs(out, exist_ok=True)
    base = 0 if split == "train" else 10_000  # disjoint seed ranges
    path = os.path.join(out, f"diffusion_{family}_N{N}_{split}.h5")
    with h5py.File(path, "w") as h:
        h.attrs.update(family=family, N=N, split=split,
                       discretisation="cell-centred FV, harmonic faces, boundary transmissibility 2k, A u = f/N^2")
        F = np.stack([np.random.default_rng(_seed("f", family, base + i)).standard_normal((N, N)) for i in range(n_inst)])
        h.create_dataset("f", data=F, compression="gzip")
        seeds = [_seed(family, base + i) for i in range(n_inst)]
        h.attrs["seeds"] = seeds
        if family.startswith("lognormal"):
            kind = family.split("_")[1]
            phis = np.stack([grf(N, np.random.default_rng(s), ell=0.05, kind=kind) for s in seeds])
            h.create_dataset("phi", data=phis, compression="gzip")
            h.attrs["sigma2"] = SIGMA2
            h.attrs["note"] = "k = exp(sqrt(sigma2) * phi); sweep sigma2 for contrast"
            ref_k = np.exp(np.sqrt(SIGMA2[-1]) * phis[0])
        else:
            gfun, mfun, sign = BINARY[family]
            masks = np.stack([mfun(N, gfun(np.random.default_rng(s))) for s in seeds]).astype(np.uint8)
            h.create_dataset("mask", data=masks, compression="gzip")
            h.attrs["contrast_exponents"] = CONTRAST_EXPONENTS
            h.attrs["sign"] = sign
            h.attrs["note"] = "k = 1 + (10^(sign*p) - 1) * mask; sweep p for contrast"
            ref_k = k_binary(masks[0], 4, sign)
        # self-check data for the Julia assembly (instance 0)
        A = assemble(ref_k)
        xg = np.random.default_rng(7).standard_normal((N, N))
        y = A @ xg.ravel(order="F")
        g = h.create_group("check")
        g.attrs["what"] = ("instance 0; lognormal: sigma2=last; binary: p=4. "
                           "y = A*vec(x) (column-major), b = rhs(f[0])")
        g.create_dataset("k", data=ref_k)
        g.create_dataset("x", data=xg)
        g.create_dataset("y", data=y.reshape((N, N), order="F"))
        g.create_dataset("b", data=rhs(F[0]).reshape((N, N), order="F"))
    return path


def build(out, sizes=(64, 128), n_train=32, n_test=16):
    fams = ["lognormal_exponential", "lognormal_gaussian"] + list(BINARY)
    paths = []
    for fam in fams:
        for N in sizes:
            for split, n in (("train", n_train), ("test", n_test)):
                paths.append(write_split(out, fam, N, split, n))
                print("wrote", paths[-1])
    return paths


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "diffusion")
    build(out)
