"""Dataset gallery and baseline curves -> results/*.png"""
import csv
import os
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as sla

import diffusion as D
import helmholtz as H

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")


def gallery():
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
    d = lambda n: h5py.File(os.path.join(DATA, "diffusion", n))
    phi = d("diffusion_lognormal_exponential_N128_train.h5")["phi"][0]
    panels = [("lognormal (sigma^2=2)", np.sqrt(2) * phi / np.log(10))]
    for fam in ("channels", "inclusions", "barriers"):
        h = d(f"diffusion_{fam}_N128_train.h5")
        panels.append((f"{fam} (p=6)", np.log10(D.k_binary(h["mask"][0], 6, int(h.attrs["sign"])))))
    for a, (t, z) in zip(ax[0], panels):
        im = a.imshow(z.T, origin="lower", cmap="viridis", extent=(0, 1, 0, 1))
        a.set_title("diffusion: " + t + "\nlog10 k"); plt.colorbar(im, ax=a, fraction=0.046)
    F = 8
    for a, kind in zip(ax[1][:3], ("layered", "smooth", "inclusions")):
        g = h5py.File(os.path.join(DATA, "helmholtz", f"helmholtz_{kind}_train.h5"))[f"freq_{F}"]
        c = g["c"][0]
        im = a.imshow(c.T, origin="lower", cmap="magma")
        a.set_title(f"Helmholtz medium: {kind}\nc(x) incl. PML"); plt.colorbar(im, ax=a, fraction=0.046)
    g = h5py.File(os.path.join(DATA, "helmholtz", "helmholtz_layered_train.h5"))[f"freq_{F}"]
    c, f = g["c"][0], g["f"][0]
    N, npml = int(g.attrs["N"]), int(g.attrs["npml"])
    A, sxsy, _ = H.assemble(c, F, N, npml)
    u = sla.spsolve(A.tocsc(), (sxsy * f).ravel(order="F")).reshape(c.shape, order="F")
    v = np.abs(u.real).max()
    im = ax[1][3].imshow(u.real.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v)
    ax[1][3].set_title(f"Re u, layered, freq={F}\n(PML absorbs at edges)"); plt.colorbar(im, ax=ax[1][3], fraction=0.046)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    os.makedirs(RES, exist_ok=True)
    plt.savefig(os.path.join(RES, "gallery.png"), dpi=110)


def curves():
    paths = [os.path.join(RES, n) for n in ("baselines_diffusion.csv", "baselines_helmholtz.csv")]
    paths = [p for p in paths if os.path.exists(p)]
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    fams = sorted({r["family"] for r in rows}, key=lambda s: (s.startswith("helm"), s))
    fig, ax = plt.subplots(1, len(fams), figsize=(3.6 * len(fams), 3.4))
    ax = np.atleast_1d(ax)
    for a, fam in zip(ax, fams):
        fails = []
        by = defaultdict(list)
        for r in rows:
            if r["family"] == fam:
                by[r["solver"]].append((float(r["value"]), float(r["iters"]) if r["iters"] != "nan" else np.nan))
        for s, pts in by.items():
            pts.sort()
            xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
            a.plot(xs, ys, "o-", label=s, ms=3)
            if np.isnan(ys).any():
                fails.append((xs[np.isnan(ys)], a.lines[-1].get_color()))
        a.set_yscale("log"); a.set_title(fam, fontsize=9)
        top = a.get_ylim()[1]
        for fx, col in fails:  # x at the top = did not converge within the cap
            a.plot(fx, np.full(len(fx), top), "x", color=col, ms=6, clip_on=False)
        a.set_xlabel("frequency (wavelengths)" if fam.startswith("helm") else "log10 contrast")
        a.set_ylabel("iterations")
        a.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "baselines.png"), dpi=110)


if __name__ == "__main__":
    import sys
    if "gallery" in sys.argv or len(sys.argv) == 1:
        gallery()
    if "curves" in sys.argv or len(sys.argv) == 1:
        curves()
