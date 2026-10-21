# Checks the Idea 2 harness on the starting programs (no LLM, free):
# Enzyme's gradient through m solver iterations vs finite differences, tuning, and
# iteration counts per contrast.
#
#   julia --project=julia julia/idea2/check_harness.jl [family] [N]

include("harness.jl")

family = length(ARGS) >= 1 ? ARGS[1] : "channels"
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64
datadir = joinpath(@__DIR__, "..", "..", "data", "diffusion")
levels = contrast_levels(datadir, family, N)
levels = levels[1:2:end]                         # every other contrast level, to keep it quick
train = load_problems(datadir, family, N, "train", 1:2, levels)
val = load_problems(datadir, family, N, "train", 5:6, levels)
m, maxit = 30, 500
println("$family N=$N: contrast levels $levels, $(length(train)) training problems, m = $m")

for (frame, file) in [("precond", "precond_jacobi_poly.jl"), ("solver", "solver_jacobi_momentum.jl")]
    println("\n=== $frame: $file")
    c = load_candidate(read(joinpath(@__DIR__, "seeds", file), String), frame)
    t0 = time()
    losses, g = Base.invokelatest(loss_and_grad, c, c.theta0, train, m)
    t = time() - t0
    println(@sprintf("loss %.3f, gradient %s (%.1f s incl. compile)", mean(losses), round.(g; sigdigits = 3), t))
    # finite differences on every parameter
    for i in eachindex(c.theta0)
        h = 1e-5 * (c.upper[i] - c.lower[i])
        θp = copy(c.theta0); θp[i] += h; θm = copy(c.theta0); θm[i] -= h
        fd = (Base.invokelatest(mean_loss, c, θp, train, m) - Base.invokelatest(mean_loss, c, θm, train, m)) / 2h
        println(@sprintf("  %-6s enzyme %+.6e   finite diff %+.6e   rel diff %.1e", c.names[i], g[i], fd,
                         abs(g[i] - fd) / max(abs(fd), 1e-12)))
    end
    before = Base.invokelatest(evaluate_iterations, c, c.theta0, val, maxit)
    t0 = time()
    best, hist, g0 = Base.invokelatest(tune, c, train, m; steps = 20)
    t = time() - t0
    after = Base.invokelatest(evaluate_iterations, c, best.θ, val, maxit)
    println(@sprintf("tuning (%.1f s): loss %.3f -> %.3f, θ %s -> %s", t, hist[1], best.L, c.theta0, round.(best.θ; digits = 3)))
    println(@sprintf("  score (mean log10 effective iterations): %.3f -> %.3f", before["score"], after["score"]))
    for k in sort(collect(keys(before["median_iterations_by_contrast"])); by = x -> parse(Float64, x))
        println(@sprintf("  contrast %-4s median effective iterations to 1e-8: %7.0f -> %7.0f   (converged %d -> %d of %d; residual at cap %.1e -> %.1e)", k,
                         before["median_iterations_by_contrast"][k], after["median_iterations_by_contrast"][k],
                         before["converged_by_contrast"][k], after["converged_by_contrast"][k], before["problems_per_contrast"][k],
                         before["median_final_residual_by_contrast"][k], after["median_final_residual_by_contrast"][k]))
    end
    rep = Base.invokelatest(gradient_report, c, best, hist, g0, train, m)
    println("  gradient check: ", rep["gradient_check"])
    println("  params: ", [(p["name"], round(p["value"]; digits = 3), p["at_bound"], round(p["relative_sensitivity"]; digits = 2)) for p in rep["params"]])
end
