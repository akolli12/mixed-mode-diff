# Times Enzyme on every problem, for comparison with TextGrad's per-step time and cost, and
# runs Enzyme gradient descent to convergence. No LLM calls; free.
#
#   julia --project=julia julia/idea1/time_enzyme.jl data/idea1_testfns data/idea1_solvers
#   julia --project=julia julia/idea1/time_enzyme.jl data/idea1_solvers --limit 3 --out /tmp/x   # quick test
#
# Writes <out>/timing.jsonl (default out: results/idea1/enzyme-timing), one row per problem:
#   compile_seconds   first Enzyme gradient call, including compilation (one-time cost)
#   gradient_seconds  median time of one Enzyme gradient after compilation
#   loss_seconds      median time of one loss evaluation
#   gd_step_seconds   median time of one gradient-descent step (gradient + step-size search)
#   gd_losses         loss after each of up to GD_MAX_STEPS gradient-descent steps, stopping
#                     early once the loss stops improving
# plus the machine it ran on, since timings depend on it.

using JSON
using LinearAlgebra
using Printf
using Statistics

include("load_program.jl")
include("scoring.jl")

const GD_MAX_STEPS = 500
const STALL_STEPS = 20       # stop when the loss improved by < STALL_TOL (relative) over this many steps
const STALL_TOL = 1e-10

# Median of repeated timings of f(), repeating until about `budget` seconds have been spent.
function median_time(f; budget=0.3, max_reps=1000)
    times = Float64[]
    while sum(times; init=0.0) < budget && length(times) < max_reps
        push!(times, @elapsed f())
    end
    return median(times)
end

function gd_to_convergence(f, θ0)
    losses = gradient_descent(f, θ0, GD_MAX_STEPS)
    for k in STALL_STEPS+1:length(losses)
        old, new = losses[k-STALL_STEPS], losses[k]
        if old - new <= STALL_TOL * max(abs(old), 1.0)
            return losses[1:k]
        end
    end
    return losses
end

function time_problem(p::LoadedProgram)
    θ0 = p.theta0
    compile = @elapsed gradient(p.f, θ0)
    gd_to_convergence(p.f, θ0)                        # warm up gradient_descent itself
    t0 = time()
    losses = gd_to_convergence(p.f, θ0)
    gd_total = time() - t0
    return Dict(
        "compile_seconds"  => compile,
        "gradient_seconds" => median_time(() -> gradient(p.f, θ0)),
        "loss_seconds"     => median_time(() -> p.f(θ0)),
        "gd_step_seconds"  => median_time(() -> gradient_descent(p.f, θ0, 1); budget=0.3) ,
        "gd_steps_run"     => length(losses) - 1,
        "gd_total_seconds" => gd_total,
        "gd_losses"        => losses,
    )
end

function main(args)
    datasets = String[]
    limit, out = typemax(Int), joinpath(@__DIR__, "..", "..", "results", "idea1", "enzyme-timing")
    i = 1
    while i <= length(args)
        if args[i] == "--limit"
            limit = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--out"
            out = args[i+1]; i += 2
        else
            push!(datasets, args[i]); i += 1
        end
    end
    mkpath(out)
    path = joinpath(out, "timing.jsonl")
    machine = Dict("cpu" => Sys.cpu_info()[1].model, "threads" => Threads.nthreads(),
                   "julia" => string(VERSION), "os" => string(Sys.KERNEL))
    open(path, "w") do io
        for dataset in datasets
            manifest = JSON.parse(read(joinpath(dataset, "manifest.json"), String))
            for (k, entry) in enumerate(manifest[1:min(limit, end)])
                file = joinpath(dataset, entry["file"])
                p = load_program(file)
                row = Base.invokelatest(time_problem, p)    # loss was defined after this loop started
                row["program"] = relpath(abspath(file), abspath(joinpath(@__DIR__, "..", "..")))
                row["meta"] = p.hidden === nothing ? nothing :
                    Dict(kk => v for (kk, v) in p.hidden if kk in ("id", "family", "n", "knobs", "f_star"))
                row["machine"] = machine
                println(io, JSON.json(row)); flush(io)
                @printf("%s %d/%d  %-40s compile %.2fs  gradient %.1e s  GD %d steps\n", dataset, k,
                        min(limit, length(manifest)), basename(file), row["compile_seconds"],
                        row["gradient_seconds"], row["gd_steps_run"])
            end
        end
    end
    println("Wrote $path")
end

main(ARGS)
