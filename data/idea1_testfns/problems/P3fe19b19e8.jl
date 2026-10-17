module P3fe19b19e8

const N = 10
const C_S = Float64[1.38379, 0.471958, 2.11851, 0.477388, 0.723591, 1.92414, 0.703331, 1.86819, 1.50498, -0.00406834]
const C_V = Float64[0.419098, -2.59372, -0.137702, -0.468103, 0.449475, 1.38682, 2.01371, 0.495905, 0.741057, 0.739595, 0.517237, 0.411436, 0.844639, -0.814352, -0.384576, -0.813666, 0.371639, 0.761093, -0.0930391, 0.280774, -0.0439529, 1.29007, 1.8011, -0.377164, -0.138736, -0.0072104, 1.31848, -1.877, -0.592623, 0.736731]
const C_K = 1
const C_C = 1
const C_A = 3.04186
const C_B = 0.872908
const THETA0 = Float64[1.12276, 1.08716, 1.3731, 1.20797, 0.504196, 0.95208, -1.13417, 0.389203, 0.270304, -0.85129]

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
