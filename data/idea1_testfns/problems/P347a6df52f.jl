module P347a6df52f

const N = 5
const C_S = Float64[-1.83749, -0.571826, -0.784413, 1.39425, -2.93372]
const C_V = Float64[-1.55042, -0.734736, 0.638089, -1.58184, 1.08975, -1.26675, 0.275464, 0.133904, -1.34495, 0.220161, 0.584559, -0.799008, -0.514515, 1.48404, 0.689661]
const C_D = Float64[100, 3.16228, 1, 31.6228, 10]
const C_W = 0.931567
const C_A = 3.97461
const C_B = -0.392313
const THETA0 = Float64[-1.736, -6.40823, 2.01949, 1.41123, -4.56254]

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
