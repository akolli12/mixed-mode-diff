module Pa91e6a394c

const N = 5
const C_S = Float64[-2.48845, 1.56227, 2.48977, -1.85636, -1.37522]
const C_V = Float64[-0.585672, 1.02932, -0.587658, -1.86146, -0.381061, -0.91276, 0.187198, 1.28323, 0.121557, 0.325597, 0.104067, -1.33292, -3.13657, -0.208063, -0.149088]
const C_K = 10
const C_C = 1
const C_A = 1.54226
const C_B = -1.18465
const THETA0 = Float64[-1.64032, -3.74574, 0.448054, 2.26325, -0.639089]

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
