# Checks that every problem in an Idea 1 dataset is in the format the harness expects
# (see load_program.jl) and that Enzyme agrees with the dataset's own check values.
#
#   julia --project=julia julia/idea1/check_dataset.jl data/idea1_testfns [data/idea1_solvers ...]
#
# For each problem in <dataset>/manifest.json:
#   - the file loads as a module defining loss and THETA0, with length(THETA0) == n
#   - loss(THETA0) matches theta0_loss in hidden/P<id>.check.json
#   - Enzyme's gradient at THETA0 matches theta0_grad in hidden/P<id>.check.json
#   - Enzyme's gradient at THETA0 matches central finite differences
#   - loss(theta_star) matches f_star in hidden/P<id>.json
# Each dataset ends with the largest Enzyme vs finite-difference error seen.

using JSON
using LinearAlgebra
using Printf

include("load_program.jl")
include("scoring.jl")

relerr(a, b) = norm(a .- b) / max(norm(b), 1e-12)

# Central finite differences, an independent check on Enzyme. Step scaled to each coordinate.
function fd_gradient(f, θ::Vector{Float64})
    g = similar(θ)
    for i in eachindex(θ)
        h = 1e-6 * max(1.0, abs(θ[i]))
        θp = copy(θ); θp[i] += h
        θm = copy(θ); θm[i] -= h
        g[i] = (f(θp) - f(θm)) / (2h)
    end
    return g
end

const FD_TOL = 1e-4      # finite differences are only accurate to roughly this
const FD_ERRORS = Float64[]

function check_problem(dataset, entry)
    p = load_program(joinpath(dataset, entry["file"]))
    # loss was defined after this loop started, so run the checks in the newest world.
    return Base.invokelatest(check_loaded, p, entry)
end

function check_loaded(p::LoadedProgram, entry)
    path = p.path
    problems = String[]
    length(p.theta0) == entry["n"] || push!(problems, "length(THETA0) = $(length(p.theta0)), manifest n = $(entry["n"])")
    p.hidden === nothing && push!(problems, "no hidden/*.json answer key")
    f0 = p.f(p.theta0)
    isfinite(f0) || push!(problems, "loss(THETA0) = $f0")
    g = gradient(p.f, p.theta0)
    all(isfinite, g) || push!(problems, "Enzyme gradient at THETA0 is not finite")
    e = relerr(g, fd_gradient(p.f, p.theta0))
    push!(FD_ERRORS, e)
    e < FD_TOL || push!(problems, @sprintf("Enzyme gradient off from finite differences by %.1e", e))
    if isfile(check_path(path))
        chk = JSON.parse(read(check_path(path), String))
        e = relerr(f0, Float64(chk["theta0_loss"]))
        e < 1e-8 || push!(problems, @sprintf("loss(THETA0) off from check file by %.1e", e))
        e = relerr(g, Float64.(collect(chk["theta0_grad"])))
        e < 1e-6 || push!(problems, @sprintf("Enzyme gradient off from check file by %.1e", e))
    else
        push!(problems, "no hidden/*.check.json")
    end
    if p.hidden !== nothing
        fs = p.f(theta_star(p))
        e = abs(fs - f_star(p)) / max(abs(f_star(p)), 1.0)
        e < 1e-6 || push!(problems, @sprintf("loss(theta_star) = %.6g but f_star = %.6g", fs, f_star(p)))
    end
    return problems
end

for dataset in ARGS
    manifest = JSON.parse(read(joinpath(dataset, "manifest.json"), String))
    bad = 0
    t0 = time()
    for (k, entry) in enumerate(manifest)
        problems = try
            check_problem(dataset, entry)
        catch err
            ["failed to load or run: " * first(sprint(showerror, err), 300)]
        end
        if !isempty(problems)
            bad += 1
            println("FAIL $(entry["file"]): ", join(problems, "; "))
        end
        if k % 25 == 0 || k == length(manifest)
            @printf("%s: checked %d/%d, %d failing, %.0fs\n", dataset, k, length(manifest), bad, time() - t0)
        end
        flush(stdout)
    end
    @printf("Enzyme vs finite differences: median rel. error %.1e, max %.1e\n",
            sort(FD_ERRORS)[cld(length(FD_ERRORS), 2)], maximum(FD_ERRORS))
    empty!(FD_ERRORS)
    println("DATASET DONE $dataset: $(length(manifest) - bad)/$(length(manifest)) OK")
end
