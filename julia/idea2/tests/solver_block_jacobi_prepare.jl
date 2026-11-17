# Test program for the optional prepare(P) (not a seed): damped block Jacobi with momentum,
#   x_{k+1} = x_k + ω B⁻¹ r_k + β (x_k - x_{k-1}),  B = the 4 x 4-cell diagonal blocks of A.
# The block inverses use `inv` (an LAPACK LU), so they are computed in prepare(P), outside
# automatic differentiation, and applied in step! only through mul!. Used by check_harness.jl.

const THETA0 = [0.8, 0.0]
const LOWER = [0.0, 0.0]
const UPPER = [1.5, 0.99]
const THETA_NAMES = ["omega", "beta"]

const B = 4                                  # block side, in cells

function prepare(P)
    N = P.N
    N % B == 0 || error("N must be a multiple of $B")
    idxs = Vector{Vector{Int}}()
    invs = Vector{Matrix{Float64}}()
    for jb in 1:(N ÷ B), ib in 1:(N ÷ B)
        idx = [i + N * (j - 1) for j in (jb - 1) * B + 1:jb * B for i in (ib - 1) * B + 1:ib * B]
        push!(idxs, idx)
        push!(invs, inv([P.A[a, b] for a in idx, b in idx]))
    end
    return (idxs = idxs, invs = invs)
end

init(P, C, θ) = (C = C, d = zeros(P.n), xprev = zeros(P.n), xold = zeros(P.n),
                 rloc = zeros(B * B), zloc = zeros(B * B))

function step!(x, r, P, S, θ)
    S.xold .= x
    for (idx, Ainv) in zip(S.C.idxs, S.C.invs)
        for a in eachindex(idx)
            S.rloc[a] = r[idx[a]]
        end
        mul!(S.zloc, Ainv, S.rloc)
        for a in eachindex(idx)
            S.d[idx[a]] = S.zloc[a]
        end
    end
    x .+= θ[1] .* S.d .+ θ[2] .* (x .- S.xprev)
    S.xprev .= S.xold
    return nothing
end
