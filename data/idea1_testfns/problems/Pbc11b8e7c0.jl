module Pbc11b8e7c0

const N = 10
const C_S = Float64[0.442912, 3.09194, 1.01437, -1.52625, 0.315709, 1.04682, -0.410658, -1.33499, -1.71133, 2.25464]
const C_V = Float64[0.747818, 1.05377, -0.437421, -1.06689, -0.843616, -0.418985, 1.21568, -1.53668, -1.31227, 1.24927, -0.123139, -1.12926, -0.279022, 0.288423, -0.561687, -0.77833, 0.643877, -1.26377, 0.419076, 0.360479, -0.313556, -1.42527, -0.974175, -0.283197, -1.95608, 0.333189, -0.0362328, -0.720092, -1.76604, 0.00271647]
const C_K = 1
const C_C = 1
const C_A = 3.92441
const C_B = 4.11592
const THETA0 = Float64[-1.97814, -1.28694, 1.53354, 0.891698, 1.89579, -0.109308, -2.67223, 0.325003, 2.09016, -0.760664]

function loss(θ::AbstractVector{T}) where {T<:Real}
    z = Vector{T}(undef, N)
    @inbounds for i in 1:N
        z[i] = θ[i]
    end
    @inbounds for r in 0:2
        vv = 0.0
        d = zero(T)
        for i in 1:N
            vv += C_V[r*N+i]^2
            d += C_V[r*N+i] * z[i]
        end
        for i in 1:N
            z[i] -= (2 * d / vv) * C_V[r*N+i]
        end
    end
    @inbounds for i in 1:N
        z[i] -= C_S[i]
    end
    s = zero(T)
    @inbounds for i in 1:N-1
        s += C_K * (z[i+1] - z[i]^2)^2 + (C_C - z[i])^2
    end
    return C_A * s + C_B
end

end # module
