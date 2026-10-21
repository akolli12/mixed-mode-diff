# Bridge between the Python search loop (python/idea2/search.py) and the Idea 2 harness.
# Started by the Python side; you do not run this by hand.
#
# Reads one JSON request per line on stdin and writes one JSON reply per line on stdout:
#   {"cmd": "setup", "family": "channels", "N": 64, "train": [1,2,3,4], "val": [5,6,7,8],
#    "levels": "every_other" | "all" | [..], "m": 30, "maxit": 500}
#   {"cmd": "evaluate", "code": "...", "frame": "precond" | "solver", "tune": true, "tune_steps": 25}
#       compile -> (tune θ with Enzyme gradients, gradient report) -> validation iterations
#   {"cmd": "test", "code": "...", "frame": "...", "theta": [...], "instances": [..], "maxit": 1000}
#       iterations on the test split at every contrast level
# Every reply has "ok"; a failed candidate reports the stage it failed at and the error.

using JSON

include("harness.jl")

const DATADIR = joinpath(@__DIR__, "..", "..", "data", "diffusion")
const CONFIG = Dict{String,Any}()
const TRAIN = DiffusionProblem[]
const VAL = DiffusionProblem[]

clean(x::AbstractFloat) = isfinite(x) ? x : nothing
clean(x::AbstractVector) = map(clean, x)
clean(x::AbstractDict) = Dict(string(k) => clean(v) for (k, v) in x)
clean(x::NamedTuple) = clean(Dict(pairs(x)))
clean(x) = x

short_error(err) = first(sprint(showerror, err), 1500) * (get(ENV, "MMD_DEBUG", "") == "1" ? sprint(Base.show_backtrace, catch_backtrace()) : "")

function pick_levels(spec, all_levels)
    spec == "all" && return all_levels
    spec == "every_other" && return all_levels[1:2:end]
    return Float64.(collect(spec))
end

function setup!(req)
    empty!(TRAIN); empty!(VAL)
    family, N = String(req["family"]), Int(req["N"])
    levels = pick_levels(req["levels"], contrast_levels(DATADIR, family, N))
    append!(TRAIN, load_problems(DATADIR, family, N, "train", Int.(req["train"]), levels))
    append!(VAL, load_problems(DATADIR, family, N, "train", Int.(req["val"]), levels))
    merge!(CONFIG, Dict("family" => family, "N" => N, "levels" => levels,
                        "m" => Int(req["m"]), "maxit" => Int(req["maxit"])))
    return Dict("ok" => true, "levels" => levels, "train_problems" => length(TRAIN),
                "val_problems" => length(VAL))
end

function evaluate(c::Candidate, tune_it::Bool, steps::Int)
    m, maxit = CONFIG["m"], CONFIG["maxit"]
    out = Dict{String,Any}("ok" => true, "names" => c.names, "theta0" => c.theta0,
                           "lower" => c.lower, "upper" => c.upper)
    stage = "tuning"
    try
        if tune_it
            best, history, g_start = tune(c, TRAIN, m; steps = steps)
            out["theta"] = best.θ
            stage = "gradient report"
            out["report"] = gradient_report(c, best, history, g_start, TRAIN, m)
        else
            stage = "training loss"
            out["theta"] = c.theta0
            out["train_loss"] = mean_loss(c, c.theta0, TRAIN, m)
        end
        stage = "validation"
        out["val"] = evaluate_iterations(c, out["theta"], VAL, maxit)
    catch err
        return Dict("ok" => false, "stage" => stage, "error" => short_error(err))
    end
    return out
end

function handle(req)
    cmd = req["cmd"]
    if cmd == "setup"
        return setup!(req)
    elseif cmd == "evaluate"
        t0 = time()
        c = try
            load_candidate(String(req["code"]), String(req["frame"]))
        catch err
            return Dict("ok" => false, "stage" => "loading the code", "error" => short_error(err))
        end
        out = Base.invokelatest(evaluate, c, Bool(req["tune"]), Int(get(req, "tune_steps", 25)))
        out["seconds"] = time() - t0
        return out
    elseif cmd == "test"
        c = load_candidate(String(req["code"]), String(req["frame"]))
        test = load_problems(DATADIR, CONFIG["family"], CONFIG["N"], "test", Int.(req["instances"]),
                             contrast_levels(DATADIR, CONFIG["family"], CONFIG["N"]))
        res = Base.invokelatest(evaluate_iterations, c, Float64.(collect(req["theta"])), test, Int(req["maxit"]))
        return merge(Dict{String,Any}("ok" => true), res)
    else
        error("unknown command: $cmd")
    end
end

for line in eachline(stdin)
    isempty(strip(line)) && continue
    reply = try
        clean(handle(JSON.parse(line)))
    catch err
        Dict("ok" => false, "stage" => "server", "error" => short_error(err))
    end
    println(stdout, JSON.json(reply))
    flush(stdout)
end
