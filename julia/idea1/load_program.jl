# Loads one Idea 1 problem in the dataset format:
#
#   data/<dataset>/problems/P<id>.jl    module P<id> defining
#                                         loss(θ::AbstractVector) -> Real   the number to minimize
#                                         THETA0::Vector{Float64}           the starting parameters
#                                       This file is exactly what the LLM is shown.
#   data/<dataset>/hidden/P<id>.json    optional answer key (theta_star, f_star, family, knobs, ...)
#                                       Used only for scoring, never shown to the LLM.

using JSON

struct LoadedProgram
    path::String
    source::String
    f::Function
    theta0::Vector{Float64}
    hidden::Union{Nothing,Dict{String,Any}}
end

hidden_path(path) = joinpath(dirname(dirname(path)), "hidden", splitext(basename(path))[1] * ".json")
check_path(path) = joinpath(dirname(dirname(path)), "hidden", splitext(basename(path))[1] * ".check.json")

function load_program(path::AbstractString)
    path = abspath(path)
    mod = Base.include(Main, path)
    mod isa Module || error("$path must contain a single `module ... end` block")
    isdefined(mod, :loss) || error("$path: module does not define `loss`")
    isdefined(mod, :THETA0) || error("$path: module does not define `THETA0`")
    hp = hidden_path(path)
    hidden = isfile(hp) ? Dict{String,Any}(JSON.parse(read(hp, String))) : nothing
    return LoadedProgram(path, read(path, String), getfield(mod, :loss),
                         Float64.(collect(getfield(mod, :THETA0))), hidden)
end

theta_star(p::LoadedProgram) = p.hidden === nothing ? nothing : Float64.(collect(p.hidden["theta_star"]))
f_star(p::LoadedProgram) = p.hidden === nothing ? nothing : Float64(p.hidden["f_star"])
