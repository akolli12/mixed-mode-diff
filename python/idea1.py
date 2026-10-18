"""
Idea 1 problem generator: disguised continuous test functions emitted as standalone
Julia files (what the LLM sees and Enzyme differentiates) plus hidden metadata
(optimum, conditioning, family) that only the evaluator reads.

Every Julia file has the same visible interface:
    module P<id>
        const N::Int                 # number of parameters
        const THETA0::Vector{Float64}
        loss(θ::AbstractVector{T}) where {T<:Real} -> T
    end
Parameters are anonymous (θ[1..N]); nothing in the file names the family.

Disguise: z = R theta - c, i.e. (1) random orthogonal mixing R = H3*H2*H1 built from
three Householder reflections (3N numbers, so the code stays short), (2) an offset c applied
*after* mixing, so theta* = R^T(z* + c) never appears as a constant in the source, (3) random loss
scale and offset so f* != 0 and f is not unit-scaled, (4) constants rounded to 6
significant digits; the Python evaluator below uses the *rounded* constants so the
hidden metadata is exact with respect to the code the LLM and Enzyme see.

Each family has a NumPy evaluator that mirrors the Julia source line for line.
Gradients for validation use the complex-step method (exact to machine precision).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

SIG = 6  # significant digits written into the Julia source


def rnd(x):
    """Round to SIG significant digits (what actually ends up in the source)."""
    return np.array([float(f"{v:.{SIG}g}") for v in np.ravel(x)]).reshape(np.shape(x))


def jl_vec(x) -> str:
    return "Float64[" + ", ".join(f"{v:.{SIG}g}" for v in np.ravel(x)) + "]"


# --------------------------------------------------------------------------------------
# shared pieces: mixing transform
# --------------------------------------------------------------------------------------
def mix(theta, S, V):
    """z = H3 H2 H1 theta - S; H_r = I - 2 v v^T / (v^T v). Works for complex theta.
    The offset is applied *after* the mixing, so theta* = R^T (z* + S) is never a constant
    that can be read off the source."""
    z = theta
    for v in V:
        z = z - (2.0 * (v @ z) / (v @ v)) * v
    return z - S


def unmix(z, S, V):
    """Inverse of mix: theta = H1 H2 H3 (z + S)."""
    y = z + S
    for v in V[::-1]:
        y = y - (2.0 * (v @ y) / (v @ v)) * v
    return y


JL_MIX = """    z = Vector{T}(undef, N)
    @inbounds for i in 1:N
        z[i] = θ[i]
    end
    @inbounds for r in 0:2
        vv = 0.0
        d = zero(T)
        for i in 1:N
            vv += C_V[r*N+i]^2
            d += C_V[r*N+i] * z[i]
        end
        for i in 1:N
            z[i] -= (2 * d / vv) * C_V[r*N+i]
        end
    end
    @inbounds for i in 1:N
        z[i] -= C_S[i]
    end
"""


# --------------------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------------------
@dataclass
class Problem:
    family: str
    n: int
    seed: int
    knob: dict
    consts: dict  # name -> rounded np array (written to source)
    theta0: np.ndarray
    body_jl: str
    f: callable = field(repr=False)
    theta_star: np.ndarray | None = None
    f_star: float | None = None
    extra: dict = field(default_factory=dict)


def _common(rng, n):
    V = rnd(rng.standard_normal((3, n)))
    s_raw = rng.uniform(-3, 3, n)                       # where the optimum roughly sits
    S = rnd(mix(s_raw, np.zeros(n), V))                 # stored constant: R s (not s itself)
    scale = float(rnd(10 ** rng.uniform(-0.3, 0.7)))
    off = float(rnd(rng.uniform(-5, 5)))
    return S, V, scale, off


def make_quadratic(n, kappa, seed):
    """f = A * 0.5 * sum_i d_i z_i^2 + B, d log-spaced in [1, kappa]. Optimum theta*=S."""
    rng = np.random.default_rng(seed)
    S, V, A, B = _common(rng, n)
    d = rnd(np.logspace(0, np.log10(kappa), n)[rng.permutation(n)] if n > 1 else np.array([1.0]))

    def f(th):
        z = mix(th, S, V)
        return A * 0.5 * np.sum(d * z * z) + B

    body = JL_MIX + """    s = zero(T)
    @inbounds for i in 1:N
        s += C_D[i] * z[i]^2
    end
    return C_A * 0.5 * s + C_B
"""
    th_star = unmix(np.zeros(n), S, V)
    th0 = th_star + _start_offset(rng, n)
    return Problem("quadratic", n, seed, {"kappa": kappa},
                   {"C_S": S, "C_V": V.ravel(), "C_D": d, "C_A": A, "C_B": B},
                   th0, body, f, theta_star=th_star, f_star=B,
                   extra={"hessian_cond_exact": float(d.max() / d.min())})


def make_logcosh(n, kappa, seed):
    """f = A * sum_i d_i w^2 log cosh(z_i / w) + B: quadratic near the optimum, linear far away.
    Step-size semantics change across regions (tests whether LLM step lengths adapt)."""
    rng = np.random.default_rng(seed)
    S, V, A, B = _common(rng, n)
    d = rnd(np.logspace(0, np.log10(kappa), n)[rng.permutation(n)] if n > 1 else np.array([1.0]))
    w = float(rnd(rng.uniform(0.3, 1.0)))

    def f(th):
        z = mix(th, S, V)
        u = z / w
        # stable log cosh: |u| + log1p(exp(-2|u|)) - log 2 (written with sqrt(u^2) so complex step works)
        au = np.sqrt(u * u)
        return A * np.sum(d * w * w * (au + np.log1p(np.exp(-2 * au)) - np.log(2.0))) + B

    body = JL_MIX + """    s = zero(T)
    @inbounds for i in 1:N
        u = abs(z[i] / C_W)
        s += C_D[i] * C_W^2 * (u + log1p(exp(-2u)) - log(2.0))
    end
    return C_A * s + C_B
"""
    th_star = unmix(np.zeros(n), S, V)
    th0 = th_star + 3 * _start_offset(rng, n)  # start in the linear region for some coordinates
    return Problem("logcosh", n, seed, {"kappa": kappa},
                   {"C_S": S, "C_V": V.ravel(), "C_D": d, "C_W": w, "C_A": A, "C_B": B},
                   th0, body, f, theta_star=th_star, f_star=B,
                   extra={"hessian_cond_exact": float(d.max() / d.min())})


def make_rosenbrock(n, b, seed):
    """Chained Rosenbrock in mixed coordinates:
    f = A * sum_{i<N} [ b (z_{i+1} - z_i^2)^2 + (1 - z_i)^2 ] + B, global min at z = 1."""
    assert n >= 2
    rng = np.random.default_rng(seed)
    S, V, A, B = _common(rng, n)
    a = 1.0  # chained Rosenbrock only has z=a*1 as a stationary point when a=1

    def f(th):
        z = mix(th, S, V)
        return A * np.sum(b * (z[1:] - z[:-1] ** 2) ** 2 + (a - z[:-1]) ** 2) + B

    body = JL_MIX + """    s = zero(T)
    @inbounds for i in 1:N-1
        s += C_K * (z[i+1] - z[i]^2)^2 + (C_C - z[i])^2
    end
    return C_A * s + C_B
"""
    zstar = np.full(n, a)
    th_star = unmix(zstar, S, V)
    # start: z0 ~ a - 1.5 + noise (the classic "start on the far side of the valley")
    z0 = zstar - 1.5 + 0.3 * rng.standard_normal(n)
    th0 = unmix(z0, S, V)
    p = Problem("rosenbrock", n, seed, {"b": b},
                {"C_S": S, "C_V": V.ravel(), "C_K": float(b), "C_C": a, "C_A": A, "C_B": B},
                th0, body, f, theta_star=th_star, f_star=B,
                extra={"note": "n>=4 also has a local minimum near z_1=-1"})
    return p


def make_logistic(n, spread, seed, lam=1e-2):
    """L2-regularized logistic regression on in-code deterministic features.
    x_ij = s_j cos(w_j t_i + phi_j), t_i = i/m; s_j log-spaced in [1, spread] -> ill-conditioning.
    Labels (+-1) are stored. Optimum found by Newton's method."""
    rng = np.random.default_rng(seed)
    S, V, A, B = _common(rng, n)
    m = max(4 * n, 40)
    sj = rnd(np.logspace(0, np.log10(spread), n)[rng.permutation(n)] if n > 1 else np.array([1.0]))
    wj = rnd(rng.uniform(1, 30, n))
    pj = rnd(rng.uniform(0, 2 * np.pi, n))
    t = np.arange(1, m + 1) / m
    X = sj * np.cos(np.outer(t, wj) + pj)
    beta = rng.standard_normal(n) / sj
    y = np.sign(X @ beta + 0.5 * rng.standard_normal(m))
    y[y == 0] = 1.0
    y = y.astype(float)

    def softplus(u):
        # stable for real input, smooth for complex-step
        if np.iscomplexobj(u):
            return np.log1p(np.exp(u))
        return np.where(u > 0, u + np.log1p(np.exp(-u)), np.log1p(np.exp(u)))

    def f(th):
        w = mix(th, S, V)
        Xc = sj * np.cos(np.outer(t, wj) + pj)
        u = -y * (Xc @ w)
        return A * (np.mean(softplus(u)) + 0.5 * lam * np.sum(w * w)) + B

    body = JL_MIX + f"""    w = z
    M = length(C_Y)
    s = zero(T)
    @inbounds for i in 1:M
        t = i / M
        dot = zero(T)
        for j in 1:N
            dot += C_SC[j] * cos(C_FR[j] * t + C_PH[j]) * w[j]
        end
        u = -C_Y[i] * dot
        s += u > 0 ? u + log1p(exp(-u)) : log1p(exp(u))
    end
    r = zero(T)
    @inbounds for j in 1:N
        r += w[j]^2
    end
    return C_A * (s / M + 0.5 * {lam} * r) + C_B
"""
    # Newton on w
    w = np.zeros(n)
    for _ in range(100):
        u = -y * (X @ w)
        sg = 1 / (1 + np.exp(-u))
        g = -(X.T @ (y * sg)) / m + lam * w
        H = (X.T * (sg * (1 - sg))) @ X / m + lam * np.eye(n)
        step = np.linalg.solve(H, g)
        w -= step
        if np.linalg.norm(step) < 1e-14 * (1 + np.linalg.norm(w)):
            break
    th_star = unmix(w, S, V)
    th0 = unmix(_start_offset(rng, n) / sj, S, V)  # scale-aware start so f(theta0) is finite and moderate
    p = Problem("logistic", n, seed, {"spread": spread, "lam": lam},
                {"C_S": S, "C_V": V.ravel(), "C_SC": sj, "C_FR": wj, "C_PH": pj, "C_Y": y, "C_A": A, "C_B": B},
                th0, body, f, theta_star=th_star)
    p.f_star = float(f(th_star))
    p.extra["hessian_cond_at_opt"] = float(np.linalg.cond(A * H))
    return p


def _start_offset(rng, n):
    u = rng.standard_normal(n)
    return u / np.linalg.norm(u) * np.sqrt(n)  # ~1 per coordinate


# --------------------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------------------
def problem_id(p: Problem) -> str:
    h = hashlib.sha1(f"{p.family}|{p.n}|{sorted(p.knob.items())}|{p.seed}".encode()).hexdigest()
    return h[:10]


def to_julia(p: Problem, pid: str) -> str:
    """The file the LLM is shown: code only. Every numeric constant (including THETA0, the
    disguise transform and the loss offset, which equals f*) lives in constants/P<id>.jl."""
    lines = [f"module P{pid}", "", f"const N = {p.n}",
             f'include(joinpath(@__DIR__, "..", "constants", "P{pid}.jl"))',
             "", "function loss(θ::AbstractVector{T}) where {T<:Real}", p.body_jl.rstrip("\n"), "end", "",
             "end # module", ""]
    return "\n".join(lines)


def to_constants(p: Problem, pid: str) -> str:
    """Numeric constants for P<id>; loaded by Julia, never shown to the LLM."""
    lines = [f"# constants for problems/P{pid}.jl (not shown to the LLM)"]
    for k, v in p.consts.items():
        if np.ndim(v) == 0:
            lines.append(f"const {k} = {float(v):.{SIG}g}")
        else:
            lines.append(f"const {k} = {jl_vec(v)}")
    lines.append(f"const THETA0 = {jl_vec(p.theta0)}")
    return "\n".join(lines) + "\n"


def write_problem(outdir: str, p: Problem, pid: str):
    for sub, text in (("problems", to_julia(p, pid)), ("constants", to_constants(p, pid))):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
        with open(os.path.join(outdir, sub, f"P{pid}.jl"), "w") as fh:
            fh.write(text)


def cs_grad(f, x, h=1e-30):
    g = np.zeros(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x), dtype=complex)
        e[i] = 1j * h
        g[i] = np.imag(f(x + e)) / h
    return g


def build_suite(outdir: str, seeds=(0, 1, 2)):
    probdir = os.path.join(outdir, "problems")
    hiddir = os.path.join(outdir, "hidden")
    os.makedirs(probdir, exist_ok=True)
    os.makedirs(hiddir, exist_ok=True)
    specs = []
    for s in seeds:
        for n in (2, 5, 10, 20, 50, 100):
            for kappa in (1, 10, 100, 1000, 10000):
                specs.append(("quadratic", make_quadratic, n, kappa, s))
                specs.append(("logcosh", make_logcosh, n, kappa, s))
        for n in (2, 5, 10, 20, 50):
            for b in (1, 10, 100):
                specs.append(("rosenbrock", make_rosenbrock, n, b, s))
            for spread in (1, 10, 100):
                specs.append(("logistic", make_logistic, n, spread, s))
    manifest = []
    for fam, mk, n, knob, s in specs:
        seed = int.from_bytes(hashlib.sha1(f"{fam}{n}{knob}{s}".encode()).digest()[:4], "little")
        p = mk(n, knob, seed)
        p.theta0 = rnd(p.theta0)  # evaluate references at exactly the THETA0 written to the source
        pid = problem_id(p)
        write_problem(outdir, p, pid)
        g_star = cs_grad(p.f, p.theta_star.astype(complex))
        f0 = float(p.f(p.theta0))
        g0 = cs_grad(p.f, p.theta0.astype(complex))
        meta = {"id": pid, "family": fam, "n": n, "seed": seed, "knobs": p.knob,
                "theta_star": p.theta_star.tolist(), "f_star": float(p.f_star),
                "f_theta0": f0, "grad_norm_theta0": float(np.linalg.norm(g0)),
                "grad_norm_at_theta_star": float(np.linalg.norm(g_star)),
                "dist_theta0_to_star": float(np.linalg.norm(p.theta0 - p.theta_star)),
                **p.extra}
        with open(os.path.join(hiddir, f"P{pid}.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        manifest.append({"id": pid, "file": f"problems/P{pid}.jl", "n": n})
        # reference values the Julia side can check its own evaluation against
        meta_check = {"theta0_loss": f0, "theta0_grad": g0.tolist()}
        with open(os.path.join(hiddir, f"P{pid}.check.json"), "w") as fh:
            json.dump(meta_check, fh)
    # public manifest: ids and sizes only, shuffled so ordering leaks nothing
    rng = np.random.default_rng(12345)
    order = rng.permutation(len(manifest))
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump([manifest[i] for i in order], fh, indent=1)
    return manifest


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "idea1_testfns")
    m = build_suite(out)
    print(f"wrote {len(m)} problems to {out}")
