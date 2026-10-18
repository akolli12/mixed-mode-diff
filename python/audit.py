"""
Independent audit of every generated file. Re-derives stored values from the files themselves
(parsing the .jl source text, re-reading HDF5) instead of trusting the generators.
Run:  python3 audit.py   (optional: pip install tree-sitter tree-sitter-julia for Julia syntax checks)
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import h5py
import numpy as np
import scipy.sparse.linalg as sla

import diffusion as D
import helmholtz as H
from idea1_solvers import Op

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
    return cond


# ---------------------------------------------------------------- Julia syntax
def julia_syntax():
    try:
        import tree_sitter as ts
        import tree_sitter_julia as tsj
    except ImportError:
        print("  (skipped: tree-sitter-julia not installed)")
        return
    parser = ts.Parser(ts.Language(tsj.language()))
    files = glob.glob(os.path.join(DATA, "idea1_*", "problems", "*.jl")) + \
        glob.glob(os.path.join(DATA, "idea1_*", "constants", "*.jl")) + \
        glob.glob(os.path.join(ROOT, "python", "pilot", "*.jl")) + \
        glob.glob(os.path.join(ROOT, "julia", "**", "*.jl"), recursive=True)
    bad = []
    for f in files:
        tree = parser.parse(open(f, "rb").read())
        if tree.root_node.has_error:
            bad.append(os.path.relpath(f, ROOT))
    check(not bad, f"Julia syntax errors in {bad[:5]}")
    print(f"  Julia syntax: {len(files)} files parsed, {len(bad)} with errors")


# ---------------------------------------------------------------- source parsing
def parse_consts(src):
    C = {}
    for m in re.finditer(r"^const (\w+) = Float64\[(.*?)\]$", src, re.M):
        C[m.group(1)] = np.array([float(x) for x in m.group(2).split(",")])
    for m in re.finditer(r"^const (\w+) = ([-+0-9.eE]+)$", src, re.M):
        C[m.group(1)] = float(m.group(2))
    return C


def jl_mix(C, th):
    N = len(th)
    z = th.astype(complex) if np.iscomplexobj(th) else th.copy()
    for r in range(3):
        v = C["C_V"][r * N:(r + 1) * N]
        z = z - (2 * (v @ z) / (v @ v)) * v
    return z - C["C_S"]


def lam_from_src(src):
    return float(re.search(r"0\.5 \* ([-+0-9.eE]+) \* r\)", src).group(1))


def eval_testfn(fam, C, src, th):
    z = jl_mix(C, th)
    A, B = C["C_A"], C["C_B"]
    if fam == "quadratic":
        return A * 0.5 * np.sum(C["C_D"] * z * z) + B
    if fam == "logcosh":
        u = np.sqrt((z / C["C_W"]) ** 2)
        return A * np.sum(C["C_D"] * C["C_W"] ** 2 * (u + np.log1p(np.exp(-2 * u)) - np.log(2.0))) + B
    if fam == "rosenbrock":
        return A * np.sum(C["C_K"] * (z[1:] - z[:-1] ** 2) ** 2 + (C["C_C"] - z[:-1]) ** 2) + B
    if fam == "logistic":
        M = len(C["C_Y"]); t = np.arange(1, M + 1) / M
        X = C["C_SC"] * np.cos(np.outer(t, C["C_FR"]) + C["C_PH"])
        u = -C["C_Y"] * (X @ z)
        sp = np.log1p(np.exp(u)) if np.iscomplexobj(u) else np.where(u > 0, u + np.log1p(np.exp(-u)), np.log1p(np.exp(u)))
        return A * (np.mean(sp) + 0.5 * lam_from_src(src) * np.sum(z * z)) + B
    raise ValueError(fam)


def csgrad(f, x, h=1e-30):
    g = np.zeros(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x), complex); e[i] = 1j * h
        g[i] = np.imag(f(x + e)) / h
    return g


def num_hessian(gf, x, h=1e-5):
    n = len(x); Hm = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h * max(1, abs(x[i]))
        Hm[:, i] = (gf(x + e) - gf(x - e)) / (2 * e[i])
    return 0.5 * (Hm + Hm.T)


def audit_suite(suite, evaluator):
    d = os.path.join(DATA, suite)
    man = json.load(open(os.path.join(d, "manifest.json")))
    ids = [m["id"] for m in man]
    files = sorted(glob.glob(os.path.join(d, "problems", "*.jl")))
    check(len(ids) == len(set(ids)) == len(files), f"{suite}: manifest/files mismatch")
    check(all(set(m) == {"id", "file", "n"} for m in man), f"{suite}: manifest leaks extra fields")
    stats = Counter(); worst = Counter()
    for m in man:
        pid = m["id"]
        src = open(os.path.join(d, m["file"])).read()
        meta = json.load(open(os.path.join(d, "hidden", f"P{pid}.json")))
        ref = json.load(open(os.path.join(d, "hidden", f"P{pid}.check.json")))
        csrc_path = os.path.join(d, "constants", f"P{pid}.jl")
        check(os.path.exists(csrc_path), f"suite {suite}: P{pid} has no constants file")
        C = parse_consts(open(csrc_path).read())
        # the LLM-visible file must contain no numeric constants at all (only N)
        shown_consts = re.findall(r"^const (\w+)", src, re.M)
        check(shown_consts == ["N"], f"P{pid}: constants visible to the LLM: {shown_consts}")
        check("Float64[" not in src, f"P{pid}: numeric array visible to the LLM")
        check(f'include(joinpath(@__DIR__, "..", "constants", "P{pid}.jl"))' in src, f"P{pid}: constants include missing")
        n = meta["n"]
        tag = f"{suite}/P{pid} ({meta['family']} n={n})"
        # structure
        check(src.startswith(f"module P{pid}\n"), f"{tag}: module name")
        check(re.search(rf"^const N = {n}$", src, re.M) is not None and m["n"] == n, f"{tag}: N mismatch")
        check(len(C["THETA0"]) == n, f"{tag}: THETA0 length")
        if "C_V" in C:
            check(len(C["C_V"]) == 3 * n and len(C["C_S"]) == n, f"{tag}: mixing constant lengths")
        for k, v in C.items():
            check(np.all(np.isfinite(v)), f"{tag}: non-finite constant {k}")
        check(f"{meta['f_star']:.6g}" not in src or meta.get("f_star_kind") == "best_known",
              f"{tag}: f* appears in the LLM-visible file")
        check("{{" not in src and "**" not in src and "np." not in src, f"{tag}: python residue in source")
        check("#" not in src.split("function loss")[1].split("\nend\n")[0] and src.count("#") == 1, f"{tag}: comment inside loss (possible hint)")
        f = lambda th: evaluator(meta, C, src, th)
        th0 = C["THETA0"]
        f0 = float(np.real(f(th0)))
        e_loss = abs(f0 - ref["theta0_loss"]) / max(1, abs(f0))
        g0 = csgrad(f, th0)
        e_grad = np.linalg.norm(g0 - ref["theta0_grad"]) / max(1, np.linalg.norm(g0))
        check(e_loss < 1e-12, f"{tag}: loss at THETA0 differs from reference ({e_loss:.1e})")
        check(e_grad < 1e-10, f"{tag}: gradient at THETA0 differs from reference ({e_grad:.1e})")
        check(abs(meta["f_theta0"] - f0) < 1e-9 * max(1, abs(f0)), f"{tag}: hidden f_theta0 wrong")
        worst["loss"] = max(worst["loss"], e_loss); worst["grad"] = max(worst["grad"], e_grad)
        ts = np.array(meta["theta_star"])
        fs = float(np.real(f(ts)))
        check(f0 > fs, f"{tag}: THETA0 not worse than theta_star")
        check(np.linalg.norm(th0 - ts) > 1e-6, f"{tag}: THETA0 == theta_star")
        if meta.get("f_star_kind") != "best_known":
            gs = csgrad(f, ts)
            rel = np.linalg.norm(gs) / max(1, np.linalg.norm(g0))
            check(rel < 1e-9, f"{tag}: gradient at theta_star not ~0 ({rel:.1e})")
            check(abs(fs - meta["f_star"]) < 1e-9 * max(1, abs(fs)), f"{tag}: f_star mismatch")
            worst["grad@opt"] = max(worst["grad@opt"], rel)
            # no constant reveals the optimum
            for k, v in C.items():
                if k == "THETA0" or np.ndim(v) == 0 or len(v) != n or n == 1:
                    continue
                dlt = v - ts
                check(not np.allclose(dlt, dlt.mean(), atol=1e-3), f"{tag}: constant {k} reveals theta_star")
            # curvature knob is real (convex families, small n)
            if meta["family"] in ("quadratic", "logcosh") and n <= 20:
                Hs = num_hessian(lambda x: csgrad(f, x), ts)
                ev = np.linalg.eigvalsh(Hs)
                kap = ev.max() / ev.min()
                check(ev.min() > 0 and abs(kap / meta["knobs"]["kappa"] - 1) < 1e-3,
                      f"{tag}: Hessian cond {kap:.4g} != kappa {meta['knobs']['kappa']}")
                stats["kappa_checked"] += 1
        else:
            check(fs <= f0 and abs(fs - meta["f_star"]) < 1e-8 * max(1, abs(fs)), f"{tag}: best_known f_star mismatch")
        stats[meta["family"]] += 1
    print(f"  {suite}: {len(man)} problems {dict(stats)}")
    print(f"    worst |loss - ref| {worst['loss']:.1e}, worst |grad - ref| {worst['grad']:.1e}"
          + (f", worst grad at optimum {worst['grad@opt']:.1e}" if worst['grad@opt'] else ""))


def eval_testfn_meta(meta, C, src, th):
    return eval_testfn(meta["family"], C, src, th)


_ops = {}


def eval_solver(meta, C, src, th):
    key = meta["id"]
    if key not in _ops:
        _ops[key] = Op(int(C["C_G"]), C["C_AMP"], C["C_F1"], C["C_F2"], C["C_P1"], C["C_P2"], C["C_R1"], C["C_R2"])
    op = _ops[key]
    fam = meta["family"]
    if fam == "solver_sor":
        from idea1_solvers import sor_loss
        return sor_loss(op, th[0], int(C["C_M"]))
    if fam == "solver_chebyshev":
        from idea1_solvers import cheb_loss
        return cheb_loss(op, th[0], th[1], int(C["C_M"]))
    if fam == "solver_richardson":
        from idea1_solvers import rich_loss
        return rich_loss(op, th)
    raise ValueError(fam)


# ---------------------------------------------------------------- diffusion
def audit_diffusion():
    if not os.path.isdir(os.path.join(DATA, "diffusion")):
        print("  (skipped: data/diffusion not generated; run make_all.sh)")
        return
    files = sorted(glob.glob(os.path.join(DATA, "diffusion", "*.h5")))
    check(len(files) == 20, f"diffusion: expected 20 files, got {len(files)}")
    seen = {}
    for p in files:
        h = h5py.File(p, "r")
        fam, N, split = h.attrs["family"], int(h.attrs["N"]), h.attrs["split"]
        tag = os.path.basename(p)
        check(os.path.basename(p) == f"diffusion_{fam}_N{N}_{split}.h5", f"{tag}: name/attrs mismatch")
        F = h["f"][:]
        check(F.shape[1:] == (N, N) and np.all(np.isfinite(F)), f"{tag}: f bad")
        check(abs(F.std() - 1) < 0.05, f"{tag}: f not unit variance")
        if fam.startswith("lognormal"):
            fld = h["phi"][:]
            check(fld.shape == F.shape and np.all(np.isfinite(fld)), f"{tag}: phi bad")
            check(np.allclose(fld.mean(axis=(1, 2)), 0, atol=1e-8) and np.allclose(fld.std(axis=(1, 2)), 1, atol=1e-8),
                  f"{tag}: phi not standardized")
        else:
            fld = h["mask"][:]
            frac = fld.mean(axis=(1, 2))
            check(set(np.unique(fld)) <= {0, 1} and frac.min() > 0.005 and frac.max() < 0.6, f"{tag}: mask degenerate")
        check(len({a.tobytes() for a in fld}) == len(fld), f"{tag}: duplicate instances")
        seen[(fam, N, split)] = {a.tobytes() for a in fld}
        g = h["check"]
        A = D.assemble(g["k"][:])
        y = A @ g["x"][:].ravel(order="F")
        check(np.allclose(y, g["y"][:].ravel(order="F"), rtol=1e-13, atol=0), f"{tag}: check A*x mismatch")
        check(np.allclose(D.rhs(F[0]), g["b"][:].ravel(order="F")), f"{tag}: check rhs mismatch")
        check(abs(A - A.T).max() < 1e-12 * abs(A).max(), f"{tag}: matrix not symmetric")
        lmin = sla.eigsh(A, k=1, sigma=0, which="LM", return_eigenvectors=False)[0] if N <= 64 else 1
        check(lmin > 0, f"{tag}: matrix not positive definite")
        # the stored check k really is instance 0 at the documented contrast
        if fam.startswith("lognormal"):
            check(np.allclose(g["k"][:], np.exp(np.sqrt(h.attrs["sigma2"][-1]) * fld[0])), f"{tag}: check k not instance 0")
        else:
            check(np.allclose(g["k"][:], D.k_binary(fld[0], 4, int(h.attrs["sign"]))), f"{tag}: check k not instance 0")
    for (fam, N, split), s in seen.items():
        if split == "train":
            check(not (s & seen[(fam, N, "test")]), f"diffusion {fam} N{N}: train/test overlap")
    print(f"  diffusion: {len(files)} files ok-checked (shapes, finiteness, SPD, A*x reference, train/test disjoint)")


# ---------------------------------------------------------------- helmholtz
def audit_helmholtz():
    if not os.path.isdir(os.path.join(DATA, "helmholtz")):
        print("  (skipped: data/helmholtz not generated; run make_all.sh)")
        return
    files = sorted(glob.glob(os.path.join(DATA, "helmholtz", "*.h5")))
    check(len(files) == 8, f"helmholtz: expected 8 files, got {len(files)}")
    seen = {}
    for p in files:
        h = h5py.File(p, "r")
        kind, split = h.attrs["medium"], h.attrs["split"]
        tag = os.path.basename(p)
        freqs = sorted(int(k.split("_")[1]) for k in h if k.startswith("freq_"))
        check(freqs == H.FREQS, f"{tag}: frequencies {freqs}")
        for F in freqs:
            g = h[f"freq_{F}"]
            N, npml = int(g.attrs["N"]), int(g.attrs["npml"])
            c, f = g["c"][:], g["f"][:]
            nt = N + 2 * npml
            check(N == int(np.ceil(H.PPW * F)) and c.shape[1:] == (nt, nt) and f.shape == c.shape, f"{tag} f{F}: shapes")
            check(np.all(np.isfinite(c)) and np.all(np.isfinite(f)), f"{tag} f{F}: non-finite")
            mins = c.min(axis=(1, 2))
            check(np.allclose(mins, 1.0), f"{tag} f{F}: min c != 1 (ppw assumption broken): {mins.min():.3g}..{mins.max():.3g}")
            check(c.max() <= 3.0 + 1e-9, f"{tag} f{F}: c max {c.max()}")
            # source peak inside the physical square
            _, _, x = H.grid(N, npml)
            for s in range(len(f)):
                i, j = np.unravel_index(np.argmax(f[s]), f[s].shape)
                check(0.15 < x[i] < 0.85 and 0.15 < x[j] < 0.85, f"{tag} f{F} inst {s}: source outside interior")
            if F == freqs[0]:
                seen[(kind, split)] = {a.tobytes() for a in f}
        A, _, _ = H.assemble(h[f"freq_{freqs[0]}/c"][0], freqs[0], int(h[f"freq_{freqs[0]}"].attrs["N"]),
                             int(h[f"freq_{freqs[0]}"].attrs["npml"]))
        g = h["check"]
        y = A @ g["x"][:].ravel(order="F")
        yr = (g["y_re"][:] + 1j * g["y_im"][:]).ravel(order="F")
        check(np.allclose(y, yr, rtol=1e-13, atol=0), f"{tag}: check A*x mismatch")
        check(abs(A - A.T).max() < 1e-10 * abs(A).max(), f"{tag}: matrix not complex-symmetric")
    for (kind, split), s in seen.items():
        if split == "train":
            check(not (s & seen[(kind, "test")]), f"helmholtz {kind}: train/test sources overlap")
    print(f"  helmholtz: {len(files)} files checked (shapes, c_min=1, sources interior, A*x reference, symmetry, disjoint splits)")


def audit_results():
    import csv
    p = os.path.join(ROOT, "results", "baselines_diffusion.csv")
    if not os.path.exists(p):
        print("  (skipped: results/baselines_diffusion.csv not generated)")
        return
    rows = list(csv.DictReader(open(p)))
    solvers = {r["solver"] for r in rows}
    check(solvers == {"CG", "Jacobi-PCG", "SGS-PCG", "AMG-PCG"}, f"baselines: unexpected solvers {solvers}")
    check(len(rows) == len({(r["family"], r["inst"], r["value"], r["solver"]) for r in rows}), "baselines: duplicate rows")
    print(f"  results: {len(rows)} diffusion rows, solvers {sorted(solvers)}")


if __name__ == "__main__":
    print("Julia syntax");          julia_syntax()
    print("Idea 1 test functions"); audit_suite("idea1_testfns", eval_testfn_meta)
    print("Idea 1 solver tasks");   audit_suite("idea1_solvers", eval_solver)
    print("Diffusion");             audit_diffusion()
    print("Helmholtz");             audit_helmholtz()
    print("Results");               audit_results()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES:")
        for m in FAIL[:40]:
            print("  -", m)
        sys.exit(1)
    print("ALL CHECKS PASSED")
