# Test program for the optional prepare(P) (not a seed): block-Jacobi preconditioning with
# exact inverses of 4 x 4-cell blocks. The inverses use `inv` (an LAPACK LU), which Enzyme
# cannot differentiate, so they are computed in prepare(P), outside automatic differentiation,
# and applied in precond! only through mul!. Used by check_harness.jl.

const THETA0 = [1.0, 0.0]
const LOWER = [0.1, -1.0]
const UPPER = [2.0, 1.0]
const THETA_NAMES = ["block_weight", "jacobi_weight"]

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

setup(P, C, θ) = (C = C, dinv = 1.0 ./ P.diag, rloc = zeros(B * B), zloc = zeros(B * B))

function precond!(z, r, P, S, θ)
    for (idx, Ainv) in zip(S.C.idxs, S.C.invs)
        for a in eachindex(idx)
            S.rloc[a] = r[idx[a]]
        end
        mul!(S.zloc, Ainv, S.rloc)
        for a in eachindex(idx)
            z[idx[a]] = θ[1] * S.zloc[a] + θ[2] * S.dinv[idx[a]] * r[idx[a]]
        end
    end
    return nothing
end
