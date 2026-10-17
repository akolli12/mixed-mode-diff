module Pb7b24513dc

const N = 10
const C_S = Float64[4.9277, -1.25838, -1.12722, -0.0146861, -0.838276, -1.30952, -0.88194, -2.7102, 2.77215, 0.312632]
const C_V = Float64[0.275338, 0.405587, -0.971221, 0.800058, -1.47606, -1.44144, -0.844191, -0.237688, 1.88511, -1.40538, -0.464482, -1.90221, 0.306326, 0.172596, -0.509152, -0.33993, 0.984084, -0.615851, -0.567173, -0.340149, -1.2181, -1.25758, -0.847993, 1.74015, -0.312132, 0.383514, -0.0921652, 1.53855, -1.19848, -0.564123]
const C_D = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const C_W = 0.849013
const C_A = 2.12243
const C_B = 3.2507
const THETA0 = Float64[4.30607, -1.69669, -4.22312, 4.72744, 1.881, 1.98124, 1.22269, 0.988114, 5.51123, 2.06524]

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
