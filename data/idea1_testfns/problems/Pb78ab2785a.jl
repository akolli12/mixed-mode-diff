module Pb78ab2785a

const N = 2
const C_S = Float64[1.73762, 1.20188]
const C_V = Float64[0.855491, -0.394971, 1.01849, -0.169774, -0.724477, -0.648695]
const C_K = 100
const C_C = 1
const C_A = 0.577558
const C_B = -2.45186
const THETA0 = Float64[-1.58339, -0.262673]

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
