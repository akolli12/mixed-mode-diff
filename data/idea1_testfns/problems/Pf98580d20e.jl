module Pf98580d20e

const N = 10
const C_S = Float64[-1.94754, -1.15331, 0.622432, 1.73149, 1.41142, 1.63162, 0.437918, -1.43259, -0.729739, -0.710055]
const C_V = Float64[-0.90314, 1.24511, -0.852619, 0.953233, 1.08254, 2.4965, -0.843992, 1.23424, -0.943447, -0.364933, -1.00058, -1.13289, -0.263745, 1.16769, 0.346777, 2.89044, -1.94477, -0.288772, 0.0635135, 1.28701, 0.595413, 0.532137, 1.28502, 0.862072, 0.559282, 0.622998, -0.207397, -1.8719, -0.153252, 0.657103]
const C_K = 1
const C_C = 1
const C_A = 2.19974
const C_B = 0.905051
const THETA0 = Float64[-2.96989, -1.2522, -1.27151, 0.0400623, 0.443936, -0.452688, 1.40362, -0.30392, -1.71348, -2.18168]

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
