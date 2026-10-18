# Idea 1: disguised test functions and frozen-solver tuning problems.
# Each problem is a standalone file  problems/P<id>.jl  defining module P<id> with
#   N, THETA0, loss(θ)
# Hidden metadata lives in hidden/P<id>.json (never show it to the LLM).

"""
    load_problem(path) -> Module

Include a problem file into a fresh module and return the problem module `P`.
Use `P.loss`, `P.THETA0`, `P.N`. Because the code is loaded at run time, call the loss from
inside functions with `problem_loss(P, θ)` (avoids world-age errors). At the REPL / top
level you can call `P.loss(θ)` and differentiate it with Enzyme directly, e.g.
    Enzyme.gradient(Reverse, P.loss, θ)
"""
function load_problem(path::AbstractString)
    host = Module(gensym(:mmd_problem))
    Base.include(host, path)
    name = Symbol(splitext(basename(path))[1])
    return getfield(host, name)
end

problem_loss(P::Module, θ) = Base.invokelatest(P.loss, θ)

_sibling(path, ext) = joinpath(dirname(dirname(path)), "hidden", splitext(basename(path))[1] * ext)

"Hidden metadata (optimum, f*, conditioning, family) for a problem file."
load_hidden(path::AbstractString) = JSON3.read(read(_sibling(path, ".json"), String))

"Reference loss and complex-step gradient at THETA0 (compare your Enzyme gradient to this)."
load_check(path::AbstractString) = JSON3.read(read(_sibling(path, ".check.json"), String))

"Paths of all problem files in a suite directory (e.g. data/idea1_testfns)."
list_problems(dir::AbstractString) =
    sort([joinpath(dir, "problems", f) for f in readdir(joinpath(dir, "problems")) if endswith(f, ".jl")])
