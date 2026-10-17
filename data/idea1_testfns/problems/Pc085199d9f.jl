module Pc085199d9f

const N = 10
const C_S = Float64[-1.55221, 0.182252, 0.252235, 2.18282, 3.56017, -3.24873, -0.362988, 0.926424, -0.0493095, -1.24709]
const C_V = Float64[-0.55564, -2.89791, -0.597323, -0.21176, -0.885283, -0.163288, 0.801025, 0.631149, -0.264231, -0.212903, -0.169892, 1.93491, -0.248315, 0.444589, 1.16148, -0.175326, 0.71129, 2.19455, -1.21868, -0.217315, 0.507628, 1.28542, -0.304721, 0.0444873, 0.807234, 0.48663, -0.692742, -1.10856, -0.703073, 0.253701]
const C_D = Float64[1.6681, 4.64159, 100, 12.9155, 1, 35.9381, 21.5443, 59.9484, 2.78256, 7.74264]
const C_A = 0.890227
const C_B = 3.13296
const THETA0 = Float64[-1.02796, -1.73829, 2.65837, 2.47645, 1.49277, -3.47242, -2.20078, -1.80387, -0.0638657, -0.23089]

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
