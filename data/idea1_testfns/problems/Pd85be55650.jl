module Pd85be55650

const N = 10
const C_S = Float64[-1.49403, -1.4574, 2.07004, -2.11455, 0.116296, 2.09864, 0.946924, -1.42186, -0.480538, 1.95624]
const C_V = Float64[-0.269708, -0.627196, -0.728382, -1.67147, -0.109703, 0.921366, 0.834271, -1.3628, 0.407509, 0.705042, -1.40138, 2.56183, 1.99925, 1.66807, -1.28857, 0.158374, 0.135409, -1.12188, -0.0263043, -0.806426, 0.127797, -0.933891, 0.997876, -0.64242, -0.95882, 0.00843571, 0.263129, -0.0806596, -0.182211, -0.144376]
const C_D = Float64[21.5443, 4.64159, 10, 46.4159, 2.15443, 464.159, 215.443, 100, 1000, 1]
const C_A = 2.83049
const C_B = -2.89651
const THETA0 = Float64[-0.559635, 0.375736, 1.33334, 2.35049, 4.28793, 0.840357, -2.53821, 0.99903, 0.235744, 1.81713]

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
