module Pcd38451a43

const N = 10
const C_S = Float64[0.854623, 2.42777, -3.4697, -1.4214, 1.55862, -0.985416, -0.604183, -1.16885, -0.81811, 1.98303]
const C_V = Float64[-0.456646, -1.00792, 1.23845, -1.2789, -0.176046, 0.401514, -0.365281, 1.62741, 0.126581, 0.402862, -1.38977, -0.521574, 0.245893, -1.09163, -0.613201, 0.422704, -1.739, -0.09275, 0.55942, -0.683882, -0.0925124, 0.384761, 1.84924, 1.22861, -0.0761617, 0.430175, 1.24419, -0.219916, 0.60243, -1.26783]
const C_D = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const C_W = 0.82464
const C_A = 4.33624
const C_B = -2.08533
const THETA0 = Float64[-4.01839, 5.64972, 8.80119, -1.71115, -2.68154, 0.943747, -3.46444, -0.872549, 0.411139, -6.52283]

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
