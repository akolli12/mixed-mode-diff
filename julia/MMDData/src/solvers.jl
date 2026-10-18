# Minimal baseline solver written to be Enzyme-friendly (generic types, in-place, no closures
# over globals). Starting point for Idea 2; richer baselines (AMG, ILU) can come from
# AlgebraicMultigrid.jl / IncompleteLU.jl / Krylov.jl.

"""
    pcg!(x, A, b; Minv=nothing, tol=1e-8, maxiter=10_000) -> iterations (or -1 if not converged)

Preconditioned conjugate gradient. `Minv` is a vector of inverse-diagonal entries (Jacobi) or
`nothing` for plain CG. Convergence on ||r|| / ||b||.
"""
function pcg!(x::AbstractVector{T}, A, b::AbstractVector; Minv = nothing, tol = 1e-8, maxiter = 10_000) where {T}
    r = b - A * x
    z = Minv === nothing ? copy(r) : Minv .* r
    p = copy(z)
    rz = dot(r, z)
    bn = norm(b)
    for it in 1:maxiter
        Ap = A * p
        α = rz / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        norm(r) <= tol * bn && return it
        z = Minv === nothing ? copy(r) : Minv .* r
        rz_new = dot(r, z)
        p .= z .+ (rz_new / rz) .* p
        rz = rz_new
    end
    return -1
end
