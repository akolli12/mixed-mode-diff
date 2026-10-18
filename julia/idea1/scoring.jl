# Ground truth from Enzyme for any program f(p): the exact gradient, scores for one
# proposed update, and an Enzyme-driven gradient-descent run for a parallel reference.

using Enzyme
using LinearAlgebra

function gradient(f, θ::Vector{Float64})
    dθ = zeros(length(θ))
    autodiff(Reverse, f, Active, Duplicated(copy(θ), dθ))
    return dθ
end

# f(θ), or NaN where the program throws (e.g. a domain error at a wild θ).
safe_eval(f, θ) = try f(θ) catch; NaN end
safe_loss(f, θ) = (v = safe_eval(f, θ); isfinite(v) ? v : Inf)

# Minimizes φ(α) = f(θ + α d) over α ≥ 0: a log-spaced grid, then golden-section refinement.
# Returns 0 when no positive α beats staying put.
function best_alpha(f, θ::Vector{Float64}, d::Vector{Float64})
    φ(α) = safe_loss(f, θ .+ α .* d)
    αs = [0.0; 10.0 .^ range(-4, 3; length=141)]
    vals = φ.(αs)
    i = argmin(vals)
    i == 1 && return 0.0
    a, b = αs[i-1], αs[min(i + 1, length(αs))]
    r = (sqrt(5) - 1) / 2
    x1, x2 = b - r * (b - a), a + r * (b - a)
    f1, f2 = φ(x1), φ(x2)
    for _ in 1:60
        if f1 < f2
            b, x2, f2 = x2, x1, f1
            x1 = b - r * (b - a); f1 = φ(x1)
        else
            a, x1, f1 = x1, x2, f2
            x2 = a + r * (b - a); f2 = φ(x2)
        end
    end
    α = (a + b) / 2
    return φ(α) <= vals[i] ? α : αs[i]
end

cosine(a, b) = (na = norm(a); nb = norm(b); na > 0 && nb > 0 ? dot(a, b) / (na * nb) : NaN)

# Scores one proposed update θ → θnew against the exact gradient at θ.
# With the answer key (xstar, fstar), also scores progress toward the true optimum.
function score_step(f, θ::Vector{Float64}, θnew::Vector{Float64}; xstar=nothing, fstar=nothing)
    g = gradient(f, θ)
    d = θnew .- θ
    f0, f1 = f(θ), safe_eval(f, θnew)
    α = norm(d) > 0 ? best_alpha(f, θ, d) : NaN
    αsd = best_alpha(f, θ, -g)
    scores = Dict{String,Any}(
        "loss_before"     => f0,
        "loss_after"      => f1,
        # The program blew up at θnew (loss infinite, NaN, or an error); not a descent.
        "diverged"        => !isfinite(f1),
        "descent"         => f1 < f0,
        # Direction: cosine between the step and −∇f (1 = steepest descent, 0 = orthogonal, −1 = uphill).
        "cos_to_neg_grad" => cosine(d, -g),
        "step_norm"       => norm(d),
        "grad_norm"       => norm(g),
        # Step size: the step's length divided by the best length along its own direction.
        # 1 = perfect, 2 = twice too far, 0.5 = half as far. Infinite when no forward move helps.
        "best_alpha"      => α,
        "step_ratio"      => α > 0 ? 1 / α : Inf,
        "loss_best_on_line" => isnan(α) ? f0 : safe_eval(f, θ .+ α .* d),
        # What an exact line search along −∇f would have reached from the same point.
        "loss_steepest_descent" => safe_eval(f, θ .- αsd .* g),
    )
    if fstar !== nothing
        scores["gap_before"] = f0 - fstar        # how far above the optimal loss
        scores["gap_after"] = f1 - fstar
    end
    if xstar !== nothing
        # Cosine between the step and the straight line to the optimum; differs from
        # cos_to_neg_grad when the problem is ill-conditioned.
        scores["cos_to_optimum"] = cosine(d, xstar .- θ)
        scores["dist_to_opt_before"] = norm(θ .- xstar)
        scores["dist_to_opt_after"] = norm(θnew .- xstar)
    end
    return scores
end

# The Enzyme reference trajectory: gradient descent with backtracking (Armijo) step sizes,
# one gradient per step, the same number of steps the LLM gets.
function gradient_descent(f, θ0::Vector{Float64}, steps::Int)
    θ = copy(θ0)
    v = f(θ)
    losses = [v]
    α = 1.0
    for _ in 1:steps
        g = gradient(f, θ)
        gg = dot(g, g)
        α *= 2.0
        while α > 1e-12 && safe_loss(f, θ .- α .* g) > v - 1e-4 * α * gg
            α /= 2.0
        end
        θ = θ .- α .* g
        v = f(θ)
        push!(losses, v)
    end
    return losses
end
