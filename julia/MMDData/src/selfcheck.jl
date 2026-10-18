# Verifies the Julia side against reference values computed by the Python generators.
# Run this first on any new machine:  using MMDData; selfcheck("path/to/data")

function _check_diffusion(path)
    h5open(path, "r") do h
        g = h["check"]
        k = pyread(g["k"]); x = pyread(g["x"]); y = pyread(g["y"])
        A = diffusion_matrix(k)
        return norm(A * vec(x) - vec(y)) / norm(vec(y))
    end
end

function _check_helmholtz(path)
    h5open(path, "r") do h
        g = h["check"]
        F = pyattr(g, "freq"); N = Int(pyattr(g, "N")); npml = Int(pyattr(g, "npml"))
        c = pyread(h["freq_$(Int(F))/c"])[1, :, :]
        x = pyread(g["x"])
        y = pyread(g["y_re"]) .+ im .* pyread(g["y_im"])
        ys = pyread(g["ys_re"]) .+ im .* pyread(g["ys_im"])
        e1 = norm(helmholtz_matrix(c, F, N, npml) * vec(x) - vec(y)) / norm(vec(y))
        e2 = norm(helmholtz_matrix(c, F, N, npml; beta = 0.5) * vec(x) - vec(ys)) / norm(vec(ys))
        return max(e1, e2)
    end
end

function _check_problem(path; fd_h = 1e-6)
    P = load_problem(path)
    ref = load_check(path)
    θ = copy(P.THETA0)
    f0 = problem_loss(P, θ)
    eloss = abs(f0 - ref.theta0_loss) / max(1.0, abs(ref.theta0_loss))
    # central finite differences vs the stored complex-step gradient (Enzyme should match the latter to ~1e-12)
    g = similar(θ)
    for i in eachindex(θ)
        e = zeros(length(θ)); e[i] = fd_h * max(1.0, abs(θ[i]))
        g[i] = (problem_loss(P, θ .+ e) - problem_loss(P, θ .- e)) / (2e[i])
    end
    gref = collect(Float64, ref.theta0_grad)
    egrad = norm(g - gref) / max(1.0, norm(gref))
    return eloss, egrad
end

"""
    selfcheck(datadir; idea1=true, verbose=true) -> Bool

Checks every diffusion and Helmholtz assembler against the stored A*x products and every
Idea 1 problem's loss/gradient at THETA0 against the Python reference.
"""
function selfcheck(datadir::AbstractString; idea1 = true, verbose = true)
    ok = true
    for f in readdir(joinpath(datadir, "diffusion"); join = true)
        endswith(f, ".h5") || continue
        e = _check_diffusion(f); ok &= e < 1e-10
        verbose && println(rpad(basename(f), 48), e < 1e-10 ? "ok  " : "FAIL", "  rel err ", e)
    end
    for f in readdir(joinpath(datadir, "helmholtz"); join = true)
        endswith(f, ".h5") || continue
        e = _check_helmholtz(f); ok &= e < 1e-10
        verbose && println(rpad(basename(f), 48), e < 1e-10 ? "ok  " : "FAIL", "  rel err ", e)
    end
    if idea1
        for suite in ("idea1_testfns", "idea1_solvers")
            d = joinpath(datadir, suite)
            isdir(d) || continue
            nbad = 0; worst = (0.0, 0.0)
            for p in list_problems(d)
                el, eg = _check_problem(p)
                bad = el > 1e-9 || eg > 1e-4
                nbad += bad
                worst = (max(worst[1], el), max(worst[2], eg))
                bad && verbose && println("  FAIL ", basename(p), " loss err ", el, " grad err ", eg)
            end
            ok &= nbad == 0
            verbose && println(rpad(suite, 48), nbad == 0 ? "ok  " : "FAIL",
                               "  worst loss err ", worst[1], ", worst FD-grad err ", worst[2])
        end
    end
    return ok
end
