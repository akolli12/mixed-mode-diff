module P8685d803b6

const N = 2
const C_S = Float64[-0.808567, -0.492817]
const C_V = Float64[-2.21605, -0.0979282, -2.44221, -0.0521497, -0.542926, -0.282846]
const C_D = Float64[10, 1]
const C_W = 0.73875
const C_A = 3.2506
const C_B = -0.797057
const THETA0 = Float64[-1.79185, 3.73991]

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
