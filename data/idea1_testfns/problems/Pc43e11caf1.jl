module Pc43e11caf1

const N = 10
const C_S = Float64[-2.2383, 0.681342, -0.434208, 2.26789, -2.03914, -0.917162, -1.75562, -3.57525, -1.70744, -1.74946]
const C_V = Float64[0.88618, -0.171022, 0.951352, -0.00767634, 0.822064, 0.627959, 0.234, -0.383354, 0.424772, -2.12086, -0.638897, 1.36166, -0.188432, -1.57962, 1.20859, -0.128673, -0.688272, -1.51764, -0.883695, 0.0772388, -0.545365, -1.21878, 0.871422, 0.428283, 0.936671, 0.57836, 0.074308, 1.00297, 1.34088, 1.73049]
const C_D = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const C_A = 4.15798
const C_B = -2.3313
const THETA0 = Float64[-2.69065, -1.73841, 2.97158, 0.459225, 1.23237, 0.317059, -0.116517, -3.2746, 1.85578, -2.21145]

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
