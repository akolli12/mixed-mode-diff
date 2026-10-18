# Per-step metrics for Idea 1 (proposal §4): direction, descent, step size.

"cos of the angle between the step d and the steepest-descent direction -g (1 downhill, 0 orthogonal, -1 uphill)."
cosine_to_descent(d::AbstractVector, g::AbstractVector) = -dot(d, g) / (norm(d) * norm(g))

"""
    best_step(f, θ, d; tmax=1e8, iters=80) -> t*

Best multiple t of the step `d` along its own line, minimising φ(t) = f(θ + t d) over t ≥ 0
(bracketing then golden section; assumes φ unimodal on the bracket). Returns 0.0 if no
t > 0 decreases f (d is not a descent direction at any tested scale).
"""
function best_step(f, θ::AbstractVector, d::AbstractVector; tmax = 1e8, iters = 80)
    φ(t) = (v = f(θ .+ t .* d); isfinite(v) ? v : Inf)
    f0 = φ(0.0)
    a, b = 0.0, 1.0
    fb = φ(b)
    if fb < f0
        while b < tmax                      # expand while still decreasing
            c = 2b; fc = φ(c)
            fc >= fb && break
            a, b, fb = b, c, fc
        end
        lo, hi = a, 2b
    else
        found = false
        for _ in 1:60                       # shrink until something decreases
            b /= 2; fb = φ(b)
            if fb < f0; found = true; break; end
        end
        found || return 0.0
        lo, hi = 0.0, 2b
    end
    gr = (sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo); x2 = lo + gr * (hi - lo)
    f1 = φ(x1); f2 = φ(x2)
    for _ in 1:iters
        if f1 < f2
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo); f1 = φ(x1)
        else
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo); f2 = φ(x2)
        end
    end
    return (lo + hi) / 2
end

"""
    step_metrics(f, g, θ, θnew) -> NamedTuple

`g` = exact gradient at θ (from Enzyme). Returns
  cosine     cos(angle(d, -g))
  descended  f(θnew) < f(θ)
  delta_f    f(θnew) - f(θ)
  t_star     best multiple of d along its own line
  step_ratio 1 / t_star  (1 = perfect length, 2 = twice too far, <1 too short;
             Inf if d is not a descent direction, i.e. cos <= 0)
"""
function step_metrics(f, g::AbstractVector, θ::AbstractVector, θnew::AbstractVector)
    d = θnew .- θ
    f0 = f(θ); f1 = f(θnew)
    # only search along d if it is a descent direction to first order; otherwise any "decrease"
    # found by shrinking the step is floating-point noise
    t = (norm(d) == 0 || dot(d, g) >= 0) ? 0.0 : best_step(f, θ, d)
    return (; cosine = norm(d) == 0 ? NaN : cosine_to_descent(d, g), descended = f1 < f0,
            delta_f = f1 - f0, t_star = t, step_ratio = t > 0 ? 1 / t : Inf)
end
