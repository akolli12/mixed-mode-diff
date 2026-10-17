module Pd4ab6c5301

const N = 10
const C_S = Float64[-0.773755, -0.0780574, 1.59255, 0.747322, -0.546083, 1.59086, -0.194899, 0.333733, 1.71254, -1.54331]
const C_V = Float64[-0.201554, 1.29451, -1.37708, 0.563215, 1.59138, -0.815503, 0.347507, -1.06868, -0.717404, 1.32963, -0.888053, -1.06093, -1.70738, 0.856794, 0.915404, 0.0155014, 2.22292, -0.39012, -0.890324, -2.39358, 0.103351, -1.53646, 0.0982706, 1.0717, 0.382641, -0.364052, 1.85842, -3.24512, -1.04299, 1.37403]
const C_K = 100
const C_C = 1
const C_A = 3.62274
const C_B = 1.75306
const THETA0 = Float64[-0.824365, 1.02473, -0.0942251, 1.05535, 0.782626, 0.01235, -0.455988, -2.39343, 0.12314, 0.997979]

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
