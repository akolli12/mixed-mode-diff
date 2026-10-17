module Pc5d34c5af4

const N = 5
const C_S = Float64[-0.666085, 1.21842, -1.18002, -0.618066, 1.47035]
const C_V = Float64[-0.21954, 1.16373, 0.372864, 2.78146, -0.151185, -0.613265, 0.880205, 0.305286, -0.245911, 1.09688, -0.216123, 1.3276, 1.48891, -0.644331, 0.571789]
const C_D = Float64[1, 1, 1, 1, 1]
const C_A = 1.05995
const C_B = -3.44671
const THETA0 = Float64[-1.75554, 0.184739, -2.38162, 0.635847, -1.31749]

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
    @inbounds for i in 1:N
        s += C_D[i] * z[i]^2
    end
    return C_A * 0.5 * s + C_B
end

end # module
