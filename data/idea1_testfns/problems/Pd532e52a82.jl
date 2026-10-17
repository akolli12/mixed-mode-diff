module Pd532e52a82

const N = 2
const C_S = Float64[1.13899, 1.22707]
const C_V = Float64[-0.586134, 0.9306, -0.0103164, 0.189137, 0.281681, 0.818245]
const C_D = Float64[1, 1000]
const C_W = 0.33892
const C_A = 4.16077
const C_B = -4.84839
const THETA0 = Float64[1.87594, -4.98427]

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
