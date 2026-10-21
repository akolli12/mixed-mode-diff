# Idea 2 harness: evaluates one LLM-written solver candidate on high-contrast diffusion.
#
# A candidate is Julia source defining its tunable numbers and one of two interfaces
# ("frames"); the LLM writes the candidate, this file does everything else:
#
#   frame "precond": the LLM writes a preconditioner; flexible CG around it is fixed.
#       THETA0, LOWER, UPPER :: Vector{Float64}   tunable numbers and their bounds
#       THETA_NAMES          :: Vector{String}    (optional) a name per number
#       setup(P, θ) -> S                          anything precomputed from the problem
#       precond!(z, r, P, S, θ)                   z = M⁻¹ r, in place
#
#   frame "solver": the LLM writes the whole iteration.
#       THETA0, LOWER, UPPER, THETA_NAMES          as above
#       init(P, θ) -> S                            state carried between iterations
#       step!(x, r, P, S, θ)                       one iteration: update x in place, given
#                                                  the current residual r = b - A x
#
# P is a DiffusionProblem (below). Training loss: mean over training instances of
# log10(‖r_m‖/‖b‖) after m iterations, differentiated with Enzyme. Evaluation: iterations
# to ‖r‖/‖b‖ ≤ TOL, per contrast level.

using Enzyme
using LinearAlgebra
using MMDData
using Printf
using SparseArrays
using Statistics

const TOL = 1e-8

# ---------------------------------------------------------------- problems

"""One linear system A x = b from the diffusion datasets, in the form candidates see."""
struct DiffusionProblem
    N::Int                       # grid is N x N cells; unknown (i, j) is i + N*(j-1)
    n::Int                       # number of unknowns, N^2
    k::Matrix{Float64}           # cell diffusion coefficients (N x N)
    tw::Matrix{Float64}          # face transmissibilities to the west/east/south/north
    te::Matrix{Float64}          # neighbour (boundary faces: 2k); A's off-diagonals are
    ts::Matrix{Float64}          # their negatives and its diagonal their sum
    tn::Matrix{Float64}
    A::SparseMatrixCSC{Float64,Int}
    diag::Vector{Float64}
    b::Vector{Float64}
    contrast::Float64            # log10 contrast (binary families) or sigma^2 (lognormal)
end

function transmissibilities(k)
    N = size(k, 1)
    hm(a, c) = 2a * c / (a + c)
    tw = similar(k); te = similar(k); ts = similar(k); tn = similar(k)
    for j in 1:N, i in 1:N
        tw[i, j] = i > 1 ? hm(k[i-1, j], k[i, j]) : 2k[i, j]
        te[i, j] = i < N ? hm(k[i+1, j], k[i, j]) : 2k[i, j]
        ts[i, j] = j > 1 ? hm(k[i, j-1], k[i, j]) : 2k[i, j]
        tn[i, j] = j < N ? hm(k[i, j+1], k[i, j]) : 2k[i, j]
    end
    return tw, te, ts, tn
end

function make_problem(ds, s, level)
    A, b, k = startswith(ds.family, "lognormal") ? diffusion_instance(ds, s; sigma2 = level) :
                                                    diffusion_instance(ds, s; p = level)
    tw, te, ts, tn = transmissibilities(k)
    return DiffusionProblem(ds.N, ds.N^2, k, tw, te, ts, tn, A, Vector(diag(A)), b, Float64(level))
end

"""Problems for the given instances (1-based) at every level of `levels`."""
function load_problems(datadir, family, N, split, instances, levels)
    ds = load_diffusion(joinpath(datadir, "diffusion_$(family)_N$(N)_$(split).h5"))
    return [make_problem(ds, s, lv) for lv in levels for s in instances]
end

contrast_levels(datadir, family, N) = begin
    ds = load_diffusion(joinpath(datadir, "diffusion_$(family)_N$(N)_train.h5"))
    startswith(family, "lognormal") ? Float64.(ds.sigma2) : Float64.(ds.contrast_exponents)
end

# ---------------------------------------------------------------- helpers for candidates

"""y = A x for the problem's matrix, written with loops so Enzyme differentiates it."""
function matvec!(y::AbstractVector, P::DiffusionProblem, x::AbstractVector)
    fill!(y, 0)
    colptr, rowval, nzval = P.A.colptr, P.A.rowval, P.A.nzval
    @inbounds for j in 1:P.n
        xj = x[j]
        for q in colptr[j]:(colptr[j+1]-1)
            y[rowval[q]] += nzval[q] * xj
        end
    end
    return y
end

# ---------------------------------------------------------------- running a candidate

"""Relative residual after m iterations of flexible CG with the candidate preconditioner."""
function run_precond(setup_f, precond_f, P::DiffusionProblem, θ, m::Int)
    S = setup_f(P, θ)
    n = P.n
    x = zeros(n); r = copy(P.b); z = zeros(n); Ap = zeros(n)
    precond_f(z, r, P, S, θ)
    p = copy(z)
    rz = dot(r, z)
    bn = norm(P.b)
    for _ in 1:m
        matvec!(Ap, P, p)
        α = rz / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        znew = zeros(n)
        precond_f(znew, r, P, S, θ)
        rznew = dot(r, znew)
        β = (rznew - dot(r, z)) / rz          # flexible (Polak–Ribière) β
        p .= znew .+ β .* p
        z = znew
        rz = rznew
    end
    return norm(r) / bn
end

"""Relative residual after m iterations of the candidate's own iteration."""
function run_solver(init_f, step_f, P::DiffusionProblem, θ, m::Int)
    S = init_f(P, θ)
    x = zeros(P.n); r = copy(P.b); Ax = zeros(P.n)
    for _ in 1:m
        step_f(x, r, P, S, θ)
        matvec!(Ax, P, x)
        r .= P.b .- Ax
    end
    return norm(r) / norm(P.b)
end

problem_loss(runner, f1, f2, P, θ, m) = log10(runner(f1, f2, P, θ, m) + 1e-30)

"""(iterations to ‖r‖/‖b‖ ≤ TOL or maxit + 1 if not reached, relative residual at the end).
A breakdown (non-finite residual) returns (maxit + 1, Inf)."""
function iterations_precond(setup_f, precond_f, P, θ, maxit)
    S = setup_f(P, θ)
    n = P.n
    x = zeros(n); r = copy(P.b); z = zeros(n); Ap = zeros(n); znew = zeros(n)
    precond_f(z, r, P, S, θ)
    p = copy(z); rz = dot(r, z); bn = norm(P.b)
    res = 1.0
    for it in 1:maxit
        matvec!(Ap, P, p)
        α = rz / dot(p, Ap)
        x .+= α .* p; r .-= α .* Ap
        res = norm(r) / bn
        isfinite(res) || return (maxit + 1, Inf)
        res <= TOL && return (it, res)
        precond_f(znew, r, P, S, θ)
        rznew = dot(r, znew)
        β = (rznew - dot(r, z)) / rz
        p .= znew .+ β .* p
        z, znew = znew, z
        rz = rznew
    end
    return (maxit + 1, res)
end

function iterations_solver(init_f, step_f, P, θ, maxit)
    S = init_f(P, θ)
    x = zeros(P.n); r = copy(P.b); Ax = zeros(P.n); bn = norm(P.b)
    res = 1.0
    for it in 1:maxit
        step_f(x, r, P, S, θ)
        matvec!(Ax, P, x)
        r .= P.b .- Ax
        res = norm(r) / bn
        isfinite(res) || return (maxit + 1, Inf)
        res <= TOL && return (it, res)
    end
    return (maxit + 1, res)
end

"""Iterations a run needed, or for one that hit maxit, the iterations it would need at
the convergence rate it showed (capped at 100 maxit; no progress or a breakdown counts
as the cap)."""
function effective_iterations(its, res, maxit)
    its <= maxit && return Float64(its)
    (isfinite(res) && res < 1) || return 100.0 * maxit
    return min(maxit * log10(TOL) / log10(res), 100.0 * maxit)
end

# ---------------------------------------------------------------- loading a candidate

struct Candidate
    frame::String
    f1::Function                 # setup / init
    f2::Function                 # precond! / step!
    theta0::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    names::Vector{String}
    mod::Module                  # keeps the loaded code alive (nothing else references the module)
end

const LOADED_MODULES = Module[]   # rooted for the life of the process, so the GC never frees them

const CANDIDATE_PRELUDE = """
using LinearAlgebra
import Main: DiffusionProblem, matvec!
"""

function load_candidate(code::String, frame::String)
    mod = Module(:Candidate)
    push!(LOADED_MODULES, mod)
    Base.include_string(mod, CANDIDATE_PRELUDE * code, "candidate.jl")
    need = frame == "precond" ? (:setup, :precond!) : (:init, :step!)
    for s in (:THETA0, :LOWER, :UPPER, need...)
        isdefined(mod, s) || error("the code must define `$s` for the $frame interface")
    end
    θ0 = Float64.(collect(getfield(mod, :THETA0)))
    lo = Float64.(collect(getfield(mod, :LOWER)))
    hi = Float64.(collect(getfield(mod, :UPPER)))
    length(θ0) == length(lo) == length(hi) || error("THETA0, LOWER and UPPER must have the same length")
    all(lo .<= hi) || error("every LOWER must be ≤ the matching UPPER")
    all(lo .<= θ0 .<= hi) || error("THETA0 must lie within [LOWER, UPPER]")
    names = isdefined(mod, :THETA_NAMES) ? String.(collect(getfield(mod, :THETA_NAMES))) :
                                           ["θ[$i]" for i in eachindex(θ0)]
    length(names) == length(θ0) || (names = ["θ[$i]" for i in eachindex(θ0)])
    return Candidate(frame, getfield(mod, need[1]), getfield(mod, need[2]), θ0, lo, hi, names, mod)
end

runner(c::Candidate) = c.frame == "precond" ? run_precond : run_solver
counter(c::Candidate) = c.frame == "precond" ? iterations_precond : iterations_solver

# ---------------------------------------------------------------- loss, gradient, tuning

"""Per-problem losses and the gradient of their mean with respect to θ (Enzyme)."""
function loss_and_grad(c::Candidate, θ::Vector{Float64}, probs, m)
    run = runner(c)
    g = zeros(length(θ))
    losses = zeros(length(probs))
    for (i, P) in enumerate(probs)
        dθ = zeros(length(θ))
        _, val = autodiff(set_runtime_activity(ReverseWithPrimal), problem_loss, Active,
                          Const(run), Const(c.f1), Const(c.f2), Const(P), Duplicated(copy(θ), dθ), Const(m))
        losses[i] = val
        g .+= dθ ./ length(probs)
    end
    return losses, g
end

mean_loss(c, θ, probs, m) = mean(problem_loss(runner(c), c.f1, c.f2, P, θ, m) for P in probs)

"""Projected Adam in normalized coordinates u = (θ - LOWER) / (UPPER - LOWER) ∈ [0, 1]."""
function tune(c::Candidate, probs, m; steps = 25, lr = 0.05)
    span = c.upper .- c.lower
    free = span .> 0
    θ = copy(c.theta0)
    losses, g = loss_and_grad(c, θ, probs, m)
    L = mean(losses)
    history = [L]
    best = (θ = copy(θ), L = L, g = copy(g), losses = losses)
    g_start = copy(g)
    isfinite(L) && all(isfinite, g) || return best, history, g_start
    u = zeros(length(θ)); u[free] .= (θ[free] .- c.lower[free]) ./ span[free]
    mom = zeros(length(θ)); vel = zeros(length(θ))
    for t in 1:steps
        gu = g .* span
        mom .= 0.9 .* mom .+ 0.1 .* gu
        vel .= 0.999 .* vel .+ 0.001 .* gu .^ 2
        step = lr .* (mom ./ (1 - 0.9^t)) ./ (sqrt.(vel ./ (1 - 0.999^t)) .+ 1e-12)
        u[free] .= clamp.(u[free] .- step[free], 0.0, 1.0)
        θ = c.lower .+ u .* span
        losses, g = loss_and_grad(c, θ, probs, m)
        L = mean(losses)
        push!(history, L)
        if !(isfinite(L) && all(isfinite, g))     # step went somewhere broken: go back, slow down
            θ = copy(best.θ); u[free] .= (θ[free] .- c.lower[free]) ./ span[free]
            lr /= 2
            losses, g = best.losses, best.g
            continue
        end
        L < best.L && (best = (θ = copy(θ), L = L, g = copy(g), losses = losses))
    end
    return best, history, g_start
end

# ---------------------------------------------------------------- gradient report

function gradient_report(c::Candidate, best, history, g_start, probs, m)
    span = c.upper .- c.lower
    gu = best.g .* span                         # gradient per unit of each parameter's range
    scale = maximum(abs.(gu); init = 0.0)
    params = []
    for i in eachindex(best.θ)
        at = span[i] == 0 ? "fixed" :
             best.θ[i] <= c.lower[i] + 0.01span[i] ? "lower" :
             best.θ[i] >= c.upper[i] - 0.01span[i] ? "upper" : nothing
        push!(params, Dict("name" => c.names[i], "value" => best.θ[i], "start" => c.theta0[i],
                           "lower" => c.lower[i], "upper" => c.upper[i],
                           "grad_per_range" => gu[i],
                           "relative_sensitivity" => scale > 0 ? abs(gu[i]) / scale : 0.0,
                           "at_bound" => at,
                           "no_effect" => abs(g_start[i] * span[i]) < 1e-9 && abs(gu[i]) < 1e-9))
    end
    by_level = Dict{String,Float64}()
    for (P, L) in zip(probs, best.losses)
        key = @sprintf("%g", P.contrast)
        by_level[key] = get(by_level, key, 0.0) + L
    end
    counts = Dict(k => count(P -> @sprintf("%g", P.contrast) == k, probs) for k in keys(by_level))
    for k in keys(by_level); by_level[k] /= counts[k]; end
    # Enzyme vs central finite differences on the most sensitive parameter.
    check = nothing
    i = argmax(abs.(gu))
    if span[i] > 0
        h = 1e-4 * span[i]
        θp = copy(best.θ); θp[i] += h; θm = copy(best.θ); θm[i] -= h
        fd = (mean_loss(c, θp, probs, m) - mean_loss(c, θm, probs, m)) / (2h)
        ad = best.g[i]
        rel = abs(ad - fd) / max(abs(fd), abs(ad), 1e-12)
        check = Dict("param" => c.names[i], "enzyme" => ad, "finite_difference" => fd,
                     "relative_difference" => rel, "agrees" => rel < 0.05)
    end
    return Dict("train_loss_start" => history[1], "train_loss_tuned" => best.L,
                "tuning_history" => history, "params" => params,
                "train_loss_by_contrast" => by_level, "gradient_check" => check)
end

# ---------------------------------------------------------------- evaluation

function evaluate_iterations(c::Candidate, θ, probs, maxit)
    count_f = counter(c)
    eff = Dict{String,Vector{Float64}}()
    resid = Dict{String,Vector{Float64}}()
    converged = Dict{String,Int}()
    t0 = time()
    total_its = 0
    for P in probs
        its, res = count_f(c.f1, c.f2, P, θ, maxit)
        key = @sprintf("%g", P.contrast)
        total_its += min(its, maxit)
        push!(get!(eff, key, Float64[]), effective_iterations(its, res, maxit))
        push!(get!(resid, key, Float64[]), res)
        converged[key] = get(converged, key, 0) + (its <= maxit)
    end
    seconds = time() - t0
    # Selection score: mean log10 effective iterations (lower is better).
    score = mean(log10(e) for v in values(eff) for e in v)
    return Dict("median_iterations_by_contrast" => Dict(k => median(v) for (k, v) in eff),
                "converged_by_contrast" => converged,
                "problems_per_contrast" => Dict(k => length(v) for (k, v) in eff),
                "median_final_residual_by_contrast" => Dict(k => median(v) for (k, v) in resid),
                "failed" => sum(length(v) for v in values(eff)) - sum(values(converged)),
                "maxit" => maxit, "score" => score,
                "seconds_per_iteration" => seconds / max(total_its, 1))
end
