"""
"Real code" tuning problems for Idea 1 (and warm-up targets for Idea 2): the code of an
iterative solver is frozen and the LLM may only change its numeric parameters.

Operator: 2-D variable-coefficient diffusion, cell-centred 5-point finite volumes on a GxG
grid, zero Dirichlet boundary, harmonic-mean face coefficients, matrix-free. The
coefficient is k(x,y) = exp(AMP * sin(2pi F1 x + P1) * sin(2pi F2 y + P2)), so contrast
= exp(2*AMP) is the difficulty knob.

Tasks (loss = log10 ||b - A x_m|| / ||b|| after a fixed number of iterations m):
  sor         theta = [omega]                      lexicographic SOR, m sweeps
  chebyshev   theta = [lambda_lo, lambda_hi]       Chebyshev semi-iteration, m steps
  richardson  theta = [alpha_1 .. alpha_m]         non-stationary Richardson step schedule

Optima are not available in closed form; we store the best value found by a multistart
local search (f_star_kind = "best_known"), plus the eigenvalue bounds of A, which give the
textbook answers (omega_opt for Poisson, [lambda_min, lambda_max], inverse Chebyshev nodes).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

from idea1 import Problem, cs_grad, problem_id, rnd, write_problem

JL_OPERATOR = """    G = C_G
    k = Matrix{Float64}(undef, G, G)
    @inbounds for j in 1:G, i in 1:G
        k[i, j] = exp(C_AMP * sin(2π * C_F1 * i / (G + 1) + C_P1) * sin(2π * C_F2 * j / (G + 1) + C_P2))
    end
    cw = similar(k); ce = similar(k); cs = similar(k); cn = similar(k)
    @inbounds for j in 1:G, i in 1:G
        cw[i, j] = i > 1 ? 2k[i-1, j] * k[i, j] / (k[i-1, j] + k[i, j]) : k[i, j]
        ce[i, j] = i < G ? 2k[i+1, j] * k[i, j] / (k[i+1, j] + k[i, j]) : k[i, j]
        cs[i, j] = j > 1 ? 2k[i, j-1] * k[i, j] / (k[i, j-1] + k[i, j]) : k[i, j]
        cn[i, j] = j < G ? 2k[i, j+1] * k[i, j] / (k[i, j+1] + k[i, j]) : k[i, j]
    end
    b = Matrix{Float64}(undef, G, G)
    @inbounds for j in 1:G, i in 1:G
        b[i, j] = 1 + 0.5 * sin(C_R1 * i / (G + 1)) * cos(C_R2 * j / (G + 1))
    end
    bnorm = sqrt(sum(abs2, b))
"""

# applies A to x, writing into y (both GxG); inlined as a nested loop where needed
JL_APPLY = """        @inbounds for j in 1:G, i in 1:G
            v = (cw[i, j] + ce[i, j] + cs[i, j] + cn[i, j]) * {x}[i, j]
            if i > 1; v -= cw[i, j] * {x}[i-1, j]; end
            if i < G; v -= ce[i, j] * {x}[i+1, j]; end
            if j > 1; v -= cs[i, j] * {x}[i, j-1]; end
            if j < G; v -= cn[i, j] * {x}[i, j+1]; end
            {y}[i, j] = v
        end
"""

JL_FINAL_RESID = """    r = zeros(T, G, G)
    @inbounds for j in 1:G, i in 1:G
        v = (cw[i, j] + ce[i, j] + cs[i, j] + cn[i, j]) * x[i, j]
        if i > 1; v -= cw[i, j] * x[i-1, j]; end
        if i < G; v -= ce[i, j] * x[i+1, j]; end
        if j > 1; v -= cs[i, j] * x[i, j-1]; end
        if j < G; v -= cn[i, j] * x[i, j+1]; end
        r[i, j] = b[i, j] - v
    end
    s = zero(T)
    @inbounds for j in 1:G, i in 1:G
        s += r[i, j]^2
    end
    return log10(sqrt(s) / bnorm)
"""


class Op:
    """NumPy mirror of JL_OPERATOR."""

    def __init__(self, G, amp, f1, f2, p1, p2, r1, r2):
        self.G = G
        I = np.arange(1, G + 1)[:, None] * np.ones((1, G))
        J = np.ones((G, 1)) * np.arange(1, G + 1)[None, :]
        k = np.exp(amp * np.sin(2 * np.pi * f1 * I / (G + 1) + p1) * np.sin(2 * np.pi * f2 * J / (G + 1) + p2))
        hm = lambda a, b: 2 * a * b / (a + b)
        cw = k.copy(); ce = k.copy(); cs = k.copy(); cn = k.copy()
        cw[1:, :] = hm(k[:-1, :], k[1:, :])
        ce[:-1, :] = hm(k[1:, :], k[:-1, :])
        cs[:, 1:] = hm(k[:, :-1], k[:, 1:])
        cn[:, :-1] = hm(k[:, 1:], k[:, :-1])
        self.k, self.cw, self.ce, self.cs, self.cn = k, cw, ce, cs, cn
        self.diag = cw + ce + cs + cn
        self.b = 1 + 0.5 * np.sin(r1 * I / (G + 1)) * np.cos(r2 * J / (G + 1))
        self.bnorm = np.sqrt(np.sum(self.b ** 2))

    def apply(self, x):
        y = self.diag * x
        y[1:, :] -= self.cw[1:, :] * x[:-1, :]
        y[:-1, :] -= self.ce[:-1, :] * x[1:, :]
        y[:, 1:] -= self.cs[:, 1:] * x[:, :-1]
        y[:, :-1] -= self.cn[:, :-1] * x[:, 1:]
        return y

    def logres(self, x):
        r = self.b - self.apply(x)
        return np.log10(np.sqrt(np.sum(r * r)) / self.bnorm)

    def sparse(self):
        G = self.G
        A = lil_matrix((G * G, G * G))
        idx = lambda i, j: i + G * j  # column-major, matches Julia vec()
        for j in range(G):
            for i in range(G):
                p = idx(i, j)
                A[p, p] = self.diag[i, j]
                if i > 0: A[p, idx(i - 1, j)] = -self.cw[i, j]
                if i < G - 1: A[p, idx(i + 1, j)] = -self.ce[i, j]
                if j > 0: A[p, idx(i, j - 1)] = -self.cs[i, j]
                if j < G - 1: A[p, idx(i, j + 1)] = -self.cn[i, j]
        return A.tocsr()

    def eig_bounds(self):
        A = self.sparse()
        lmax = eigsh(A, k=1, which="LA", v0=np.ones(A.shape[0]), return_eigenvectors=False)[0]
        lmin = eigsh(A, k=1, sigma=0, which="LM", v0=np.ones(A.shape[0]), return_eigenvectors=False)[0]
        return float(lmin), float(lmax)


def _op_consts(rng, G, amp):
    c = {"C_G": G, "C_AMP": float(amp),
         "C_F1": float(rnd(rng.uniform(0.5, 2.5))), "C_F2": float(rnd(rng.uniform(0.5, 2.5))),
         "C_P1": float(rnd(rng.uniform(0, 6.28))), "C_P2": float(rnd(rng.uniform(0, 6.28))),
         "C_R1": float(rnd(rng.uniform(2, 9))), "C_R2": float(rnd(rng.uniform(2, 9)))}
    op = Op(G, amp, c["C_F1"], c["C_F2"], c["C_P1"], c["C_P2"], c["C_R1"], c["C_R2"])
    return c, op


# ---------------- SOR ----------------
def sor_loss(op, omega, m):
    G = op.G
    x = np.zeros((G, G), dtype=np.result_type(omega, float))
    for _ in range(m):
        for j in range(G):
            for i in range(G):
                s = op.b[i, j]
                if i > 0: s = s + op.cw[i, j] * x[i - 1, j]
                if i < G - 1: s = s + op.ce[i, j] * x[i + 1, j]
                if j > 0: s = s + op.cs[i, j] * x[i, j - 1]
                if j < G - 1: s = s + op.cn[i, j] * x[i, j + 1]
                x[i, j] = (1 - omega) * x[i, j] + omega * s / op.diag[i, j]
    return op.logres(x)


JL_SOR = JL_OPERATOR + """    ω = θ[1]
    x = zeros(T, G, G)
    for sweep in 1:C_M
        @inbounds for j in 1:G, i in 1:G
            s = T(b[i, j])
            if i > 1; s += cw[i, j] * x[i-1, j]; end
            if i < G; s += ce[i, j] * x[i+1, j]; end
            if j > 1; s += cs[i, j] * x[i, j-1]; end
            if j < G; s += cn[i, j] * x[i, j+1]; end
            x[i, j] = (1 - ω) * x[i, j] + ω * s / (cw[i, j] + ce[i, j] + cs[i, j] + cn[i, j])
        end
    end
""" + JL_FINAL_RESID


def make_sor(G, amp, m, seed):
    rng = np.random.default_rng(seed)
    c, op = _op_consts(rng, G, amp)
    c["C_M"] = m
    f = lambda th: sor_loss(op, th[0], m)
    res = minimize_scalar(lambda w: sor_loss(op, w, m), bounds=(0.05, 1.99), method="bounded",
                          options={"xatol": 1e-8})
    p = Problem("solver_sor", 1, seed, {"amp": amp, "contrast": float(np.exp(2 * amp)), "m": m},
                c, np.array([1.0]), JL_SOR, f, theta_star=np.array([res.x]), f_star=float(res.fun))
    return p, op


# ---------------- Chebyshev ----------------
def cheb_loss(op, lo, hi, m):
    G = op.G
    dt = np.result_type(lo, hi, float)
    th = (hi + lo) / 2
    de = (hi - lo) / 2
    sigma = th / de
    rho = 1 / sigma
    x = np.zeros((G, G), dtype=dt)
    r = op.b.astype(dt)
    d = r / th
    for _ in range(m):
        x = x + d
        r = r - op.apply(d)
        rho_new = 1 / (2 * sigma - rho)
        d = rho_new * rho * d + (2 * rho_new / de) * r
        rho = rho_new
    return op.logres(x)


JL_CHEB = JL_OPERATOR + """    lo = θ[1]; hi = θ[2]
    th = (hi + lo) / 2
    de = (hi - lo) / 2
    sigma = th / de
    rho = 1 / sigma
    x = zeros(T, G, G)
    r = Matrix{T}(b)
    d = r ./ th
    Ad = zeros(T, G, G)
    for it in 1:C_M
        x .+= d
""" + JL_APPLY.format(x="d", y="Ad") + """        r .-= Ad
        rho_new = 1 / (2 * sigma - rho)
        d .= rho_new * rho .* d .+ (2 * rho_new / de) .* r
        rho = rho_new
    end
""" + JL_FINAL_RESID


def make_cheb(G, amp, m, seed):
    rng = np.random.default_rng(seed)
    c, op = _op_consts(rng, G, amp)
    c["C_M"] = m
    lmin, lmax = op.eig_bounds()
    f = lambda th: cheb_loss(op, th[0], th[1], m)
    best = None
    for x0 in ([lmin, lmax], [0.5 * lmin, 1.05 * lmax], [2 * lmin, lmax]):
        r = minimize(lambda t: f(t) if t[0] < t[1] and t[0] > 0 else 50.0, x0, method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    th0 = np.array(rnd([10 * lmin, 1.3 * lmax]))
    p = Problem("solver_chebyshev", 2, seed, {"amp": amp, "contrast": float(np.exp(2 * amp)), "m": m},
                c, th0, JL_CHEB, f, theta_star=best.x, f_star=float(best.fun),
                extra={"lambda_min": lmin, "lambda_max": lmax,
                       "loss_at_true_bounds": float(f(np.array([lmin, lmax])))})
    return p, op


# ---------------- Richardson schedule ----------------
def rich_loss(op, alphas):
    G = op.G
    x = np.zeros((G, G), dtype=np.result_type(alphas, float))
    for a in alphas:
        x = x + a * (op.b - op.apply(x))
    return op.logres(x)


JL_RICH = JL_OPERATOR + """    x = zeros(T, G, G)
    Ax = zeros(T, G, G)
    for it in 1:N
""" + JL_APPLY.format(x="x", y="Ax") + """        @inbounds for j in 1:G, i in 1:G
            x[i, j] += θ[it] * (b[i, j] - Ax[i, j])
        end
    end
""" + JL_FINAL_RESID


def make_rich(G, amp, m, seed):
    rng = np.random.default_rng(seed)
    c, op = _op_consts(rng, G, amp)
    lmin, lmax = op.eig_bounds()
    f = lambda th: rich_loss(op, th)
    # textbook answer: reciprocal Chebyshev nodes on [lmin, lmax]
    kk = np.arange(1, m + 1)
    nodes = (lmax + lmin) / 2 + (lmax - lmin) / 2 * np.cos((2 * kk - 1) * np.pi / (2 * m))
    cheb_alphas = 1 / nodes
    best = None
    for x0 in (cheb_alphas, np.full(m, 1 / lmax), np.full(m, 2 / (lmin + lmax))):
        r = minimize(f, x0, jac=lambda t: cs_grad(f, t.astype(complex)), method="L-BFGS-B",
                     options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-12})
        if best is None or r.fun < best.fun:
            best = r
    th0 = np.array(rnd(np.full(m, 1 / (op.diag.max() * 2))))  # Gershgorin-safe constant step
    p = Problem("solver_richardson", m, seed, {"amp": amp, "contrast": float(np.exp(2 * amp))},
                c, th0, JL_RICH, f, theta_star=best.x, f_star=float(best.fun),
                extra={"lambda_min": lmin, "lambda_max": lmax,
                       "loss_at_chebyshev_schedule": float(f(cheb_alphas)),
                       "chebyshev_schedule": cheb_alphas.tolist()})
    return p, op


def build_suite(outdir, seeds=(0, 1, 2), G=16):
    probdir = os.path.join(outdir, "problems"); os.makedirs(probdir, exist_ok=True)
    hiddir = os.path.join(outdir, "hidden"); os.makedirs(hiddir, exist_ok=True)
    manifest = []
    for s in seeds:
        for amp in (0.0, 1.0, 2.0, 3.0):  # contrast 1, ~7, ~55, ~400
            jobs = [(make_sor, (G, amp, 20)), (make_cheb, (G, amp, 30))]
            jobs += [(make_rich, (G, amp, mm)) for mm in (4, 8, 16)]
            for mk, args in jobs:
                seed = int.from_bytes(hashlib.sha1(f"{mk.__name__}{args}{s}".encode()).digest()[:4], "little")
                p, op = mk(*args, seed)
                pid = problem_id(p)
                write_problem(outdir, p, pid)
                f0 = float(p.f(p.theta0))
                g0 = cs_grad(p.f, p.theta0.astype(complex))
                meta = {"id": pid, "family": p.family, "n": p.n, "seed": seed, "knobs": p.knob,
                        "theta_star": np.atleast_1d(p.theta_star).tolist(), "f_star": p.f_star,
                        "f_star_kind": "best_known", "f_theta0": f0,
                        "grad_norm_theta0": float(np.linalg.norm(g0)), **p.extra}
                json.dump(meta, open(os.path.join(hiddir, f"P{pid}.json"), "w"), indent=1)
                json.dump({"theta0_loss": f0, "theta0_grad": g0.tolist()},
                          open(os.path.join(hiddir, f"P{pid}.check.json"), "w"))
                manifest.append({"id": pid, "file": f"problems/P{pid}.jl", "n": p.n})
                print(f"{p.family:18s} n={p.n:2d} amp={args[1]} f0={f0:7.3f} f*={p.f_star:7.3f}")
    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "idea1_solvers")
    build_suite(out)
