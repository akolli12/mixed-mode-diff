# Classical derivative-free baselines: the fair comparison for the LLM optimizers, which
# also see only loss values (no gradients). No LLM calls; free.
#
#   julia --project=julia julia/idea1/dfo_baselines.jl data/idea1_testfns data/idea1_solvers
#
# For every problem, starting from THETA0, records the best loss found after each loss
# evaluation (index 0 = THETA0 itself, then evaluations 1..BUDGET) for
#   es    (1+1)-evolution strategy: random Gaussian steps with the 1/5th success rule for
#         the step size; ES_SEEDS independent runs
#   nm    Nelder–Mead with the standard coefficients; its first n evaluations build the
#         initial simplex, so for n >= BUDGET it never gets past that
# so each LLM method can be compared at an equal number of loss evaluations (TextGrad and
# direct prompting: 1 per step; OPRO: 4 per step).
#
# Writes results/idea1/dfo/dfo.jsonl, one row per problem.

using JSON
using LinearAlgebra
using Printf
using Random

include("load_program.jl")

const BUDGET = 100
const ES_SEEDS = 5

safe(f, θ) = (v = try f(θ) catch; NaN end; isfinite(v) ? v : Inf)

# Initial per-coordinate step size, scaled to the size of the starting point.
step0(θ0) = 0.1 * max(1.0, norm(θ0) / sqrt(length(θ0)))

function es_run(f, θ0, rng)
    x, fx, σ = copy(θ0), safe(f, θ0), step0(θ0)
    best = [fx]
    for _ in 1:BUDGET
        y = x .+ σ .* randn(rng, length(x))
        fy = safe(f, y)
        if fy < fx
            x, fx, σ = y, fy, σ * exp(1 / 3)
        else
            σ *= exp(-1 / 12)                 # 1/5th rule: success rate 1/5 keeps σ constant
        end
        push!(best, min(best[end], fy))
    end
    return best
end

function nm_run(f, θ0)
    n = length(θ0)
    best = [safe(f, θ0)]
    evals = Ref(0)
    function F(x)
        v = safe(f, x)
        evals[] += 1
        evals[] <= BUDGET && push!(best, min(best[end], v))
        return v
    end
    # Initial simplex: THETA0 plus a step along each axis (5% of the coordinate, or 0.05).
    simplex = [copy(θ0)]
    for i in 1:n
        x = copy(θ0)
        x[i] += θ0[i] == 0 ? 0.05 : 0.05 * θ0[i]
        push!(simplex, x)
    end
    vals = [best[1]; [F(simplex[i]) for i in 2:n+1 if evals[] < BUDGET]]
    length(vals) < n + 1 && return fill_to_budget(best)
    while evals[] < BUDGET
        order = sortperm(vals)
        simplex, vals = simplex[order], vals[order]
        c = sum(simplex[1:n]) ./ n
        xr = c .+ (c .- simplex[end]); fr = F(xr)
        if fr < vals[1]
            xe = c .+ 2 .* (c .- simplex[end]); fe = evals[] < BUDGET ? F(xe) : Inf
            simplex[end], vals[end] = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < vals[n]
            simplex[end], vals[end] = xr, fr
        else
            xc = fr < vals[end] ? c .+ 0.5 .* (xr .- c) : c .+ 0.5 .* (simplex[end] .- c)
            fc = evals[] < BUDGET ? F(xc) : Inf
            if fc < min(fr, vals[end])
                simplex[end], vals[end] = xc, fc
            else                               # shrink toward the best vertex
                for i in 2:n+1
                    evals[] >= BUDGET && break
                    simplex[i] = simplex[1] .+ 0.5 .* (simplex[i] .- simplex[1])
                    vals[i] = F(simplex[i])
                end
            end
        end
    end
    return fill_to_budget(best)
end

fill_to_budget(best) = [best; fill(best[end], BUDGET + 1 - length(best))]

function run_problem(p::LoadedProgram)
    es = [es_run(p.f, p.theta0, Xoshiro(seed)) for seed in 1:ES_SEEDS]
    return Dict{String,Any}("es" => es, "nm" => nm_run(p.f, p.theta0))
end

function main(args)
    out = joinpath(@__DIR__, "..", "..", "results", "idea1", "dfo")
    mkpath(out)
    path = joinpath(out, "dfo.jsonl")
    repo = abspath(joinpath(@__DIR__, "..", ".."))
    open(path, "w") do io
        for dataset in args
            manifest = JSON.parse(read(joinpath(dataset, "manifest.json"), String))
            for (k, entry) in enumerate(manifest)
                file = joinpath(dataset, entry["file"])
                p = load_program(file)
                row = Base.invokelatest(run_problem, p)     # loss was defined after this loop started
                row["program"] = relpath(abspath(file), repo)
                row["budget"] = BUDGET
                row["meta"] = p.hidden === nothing ? nothing :
                    Dict(kk => v for (kk, v) in p.hidden if kk in ("id", "family", "n", "knobs", "f_star"))
                println(io, JSON.json(row)); flush(io)
                if k % 25 == 0 || k == length(manifest)
                    @printf("%s: %d/%d problems\n", dataset, k, length(manifest))
                end
            end
        end
    end
    println("Wrote $path")
end

main(ARGS)
