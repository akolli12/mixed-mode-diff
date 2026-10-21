# Starting program for the "precond" frame: a polynomial in the Jacobi-scaled operator,
#   M⁻¹ r = Σ_j c_j (D⁻¹A)^j D⁻¹ r.
# With THETA0 = [1, 0, 0] it is exactly Jacobi-preconditioned CG.

const THETA0 = [1.0, 0.0, 0.0]
const LOWER = [-3.0, -3.0, -3.0]
const UPPER = [3.0, 3.0, 3.0]
const THETA_NAMES = ["c0", "c1", "c2"]

setup(P, θ) = (dinv = 1.0 ./ P.diag, w = zeros(P.n))

function precond!(z, r, P, S, θ)
    v = S.dinv .* r
    z .= θ[1] .* v
    for j in 2:length(θ)
        matvec!(S.w, P, v)
        v = S.dinv .* S.w
        z .+= θ[j] .* v
    end
    return nothing
end
