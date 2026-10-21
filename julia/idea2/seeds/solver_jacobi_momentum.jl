# Starting program for the "solver" frame: damped Jacobi with momentum,
#   x_{k+1} = x_k + ω D⁻¹ r_k + β (x_k - x_{k-1}).

const THETA0 = [0.8, 0.0]
const LOWER = [0.0, 0.0]
const UPPER = [2.0, 0.99]
const THETA_NAMES = ["omega", "beta"]

init(P, θ) = (dinv = 1.0 ./ P.diag, xprev = zeros(P.n), xold = zeros(P.n))

function step!(x, r, P, S, θ)
    S.xold .= x
    x .+= θ[1] .* S.dinv .* r .+ θ[2] .* (x .- S.xprev)
    S.xprev .= S.xold
    return nothing
end
