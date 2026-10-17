module P4f470be215

const N = 10
const C_S = Float64[-0.178797, 0.244496, -0.501125, 2.30969, 0.598633, -4.74735, 0.413871, 1.15955, 0.726397, 0.83024]
const C_V = Float64[1.04478, 0.741717, 1.96382, 0.272314, -0.231368, 1.49163, 0.637268, 0.473689, 0.737829, -0.6636, -1.50725, 2.82703, 0.906802, 0.879888, -2.14406, 1.04959, 0.349024, 0.994485, 0.212308, 0.583508, -1.02881, -0.564918, -0.92861, -0.143449, -0.867376, 0.109169, -1.10979, 0.721689, -0.245692, 0.0632334]
const C_K = 100
const C_C = 1
const C_A = 1.5401
const C_B = 1.95111
const THETA0 = Float64[0.652282, 2.13897, 2.02628, 2.69586, -0.423533, -3.16982, 1.62885, 1.56508, 1.72625, -0.113792]

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
