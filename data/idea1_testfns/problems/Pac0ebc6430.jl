module Pac0ebc6430

const N = 2
const C_S = Float64[-1.19664, 2.50843]
const C_V = Float64[1.29228, -1.85011, 0.930769, 1.45154, 1.14156, 0.492723]
const C_D = Float64[10, 1]
const C_W = 0.408269
const C_A = 2.67576
const C_B = 1.85433
const THETA0 = Float64[2.75617, -0.805892]

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
