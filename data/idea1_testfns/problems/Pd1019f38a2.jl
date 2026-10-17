module Pd1019f38a2

const N = 5
const C_S = Float64[1.03401, -2.17419, 1.07347, 1.59036, -1.18382]
const C_V = Float64[0.559167, -1.59594, -1.56526, -0.595391, -1.18746, 1.28573, -0.0137686, 0.649105, 1.66321, -0.588727, 2.34849, 0.759069, 0.419465, 0.00788463, 1.15291]
const C_D = Float64[1000, 5.62341, 31.6228, 1, 177.828]
const C_W = 0.502409
const C_A = 3.63892
const C_B = -4.11839
const THETA0 = Float64[-1.85376, 1.71567, 7.72902, -0.37179, 2.24411]

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
