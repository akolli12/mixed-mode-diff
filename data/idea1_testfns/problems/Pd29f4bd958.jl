module Pd29f4bd958

const N = 2
const C_S = Float64[1.74724, -1.69076]
const C_V = Float64[0.191647, -0.238688, -1.25568, 0.102631, -0.151513, -0.382649]
const C_D = Float64[10000, 1]
const C_W = 0.557486
const C_A = 0.589352
const C_B = 3.66664
const THETA0 = Float64[-3.12721, -5.39846]

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
