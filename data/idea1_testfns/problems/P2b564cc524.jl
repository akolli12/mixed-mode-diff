module P2b564cc524

const N = 5
const C_S = Float64[-0.392993, -0.705729, -0.258321, 0.426833, 1.85172]
const C_V = Float64[-0.204009, 0.482044, -0.0765515, 2.43816, 1.24739, 0.95641, 0.149913, 0.198626, 1.09634, -0.119037, -1.59576, 0.557977, -0.145862, 0.0364805, 2.09909]
const C_D = Float64[10, 3.16228, 1, 100, 31.6228]
const C_W = 0.646148
const C_A = 2.87871
const C_B = -2.19155
const THETA0 = Float64[-2.22221, -5.33391, -3.32705, -0.570185, -2.24854]

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
        u = abs(z[i] / C_W)
        s += C_D[i] * C_W^2 * (u + log1p(exp(-2u)) - log(2.0))
    end
    return C_A * s + C_B
end

end # module
