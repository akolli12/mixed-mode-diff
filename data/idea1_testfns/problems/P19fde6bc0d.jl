module P19fde6bc0d

const N = 5
const C_S = Float64[-0.426895, 3.93812, -0.0441577, 0.4282, 1.5725]
const C_V = Float64[1.08865, 1.14976, -2.05053, 0.394966, -0.648906, 0.265859, 0.117314, -0.138831, 0.247033, 1.75463, -0.51855, -0.945884, -0.532694, -0.440927, 0.802092]
const C_K = 1
const C_C = 1
const C_A = 0.802397
const C_B = -3.83356
const THETA0 = Float64[-3.18175, 0.850914, -0.595428, -1.53106, -0.938913]

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
