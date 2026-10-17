module Pb3bf368f4e

const N = 2
const C_S = Float64[-3.04038, -1.18989]
const C_V = Float64[1.31039, 0.131781, 1.47765, 0.0418476, 0.87274, -1.04213]
const C_K = 100
const C_C = 1
const C_A = 2.17853
const C_B = -4.96509
const THETA0 = Float64[-2.06875, -3.19414]

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
