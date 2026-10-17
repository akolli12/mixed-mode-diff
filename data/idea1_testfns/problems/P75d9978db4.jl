module P75d9978db4

const N = 2
const C_S = Float64[-0.960459, -0.17925]
const C_V = Float64[-0.315407, -1.01133, 0.373042, -1.2448, -1.52689, -1.80448]
const C_D = Float64[10000, 1]
const C_W = 0.367933
const C_A = 0.960149
const C_B = -3.22609
const THETA0 = Float64[2.91624, 4.08894]

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
