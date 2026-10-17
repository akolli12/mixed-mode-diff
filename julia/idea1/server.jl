# Bridge between the Python TextGrad harness and one Julia problem file.
# Started automatically by python/idea1/oracle.py; you do not run this by hand.
#
#   julia --project=julia julia/idea1/server.jl data/idea1_testfns/problems/P<id>.jl
#
# The problem format is described in load_program.jl.
#
# Reads one JSON request per line on stdin and writes one JSON reply per line on stdout:
#   {"cmd": "info"}                                      source, THETA0, loss(THETA0), answer-key metadata
#   {"cmd": "loss",  "theta": [...]}                     loss(θ)
#   {"cmd": "score", "theta": [...], "theta_new": [...]} Enzyme scores for one update
#   {"cmd": "gd",    "steps": 10}                        Enzyme gradient-descent losses

using JSON

include("load_program.jl")
include("scoring.jl")

const PROG = load_program(ARGS[1])
const XSTAR = theta_star(PROG)
const FSTAR = f_star(PROG)

# Answer-key fields the harness logs for grouping results (never sent to the LLM).
metadata(p) = p.hidden === nothing ? nothing :
    Dict(k => v for (k, v) in p.hidden if k in ("id", "family", "n", "knobs", "f_star", "f_star_kind"))

# JSON has no NaN or Inf; send them as null.
clean(x::AbstractFloat) = isfinite(x) ? x : nothing
clean(x::AbstractVector) = map(clean, x)
clean(x::AbstractDict) = Dict(k => clean(v) for (k, v) in x)
clean(x) = x

floats(v) = Float64.(collect(v))

function handle(req)
    cmd = req["cmd"]
    if cmd == "info"
        return Dict("source" => PROG.source, "theta0" => PROG.theta0,
                    "loss0" => PROG.f(PROG.theta0), "meta" => metadata(PROG))
    elseif cmd == "loss"
        return Dict("loss" => PROG.f(floats(req["theta"])))
    elseif cmd == "score"
        return score_step(PROG.f, floats(req["theta"]), floats(req["theta_new"]); xstar=XSTAR, fstar=FSTAR)
    elseif cmd == "gd"
        return Dict("losses" => gradient_descent(PROG.f, PROG.theta0, Int(req["steps"])))
    else
        error("unknown command: $cmd")
    end
end

for line in eachline(stdin)
    isempty(strip(line)) && continue
    reply = try
        clean(handle(JSON.parse(line)))
    catch err
        Dict("error" => sprint(showerror, err))
    end
    println(stdout, JSON.json(reply))
    flush(stdout)
end
