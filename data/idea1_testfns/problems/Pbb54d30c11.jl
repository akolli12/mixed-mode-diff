module Pbb54d30c11

const N = 5
const C_S = Float64[2.69097, 1.83169, -0.746527, -0.0336042, 0.297318]
const C_V = Float64[0.681704, -1.21914, -1.65372, -1.88916, -1.12994, 0.859714, -0.130133, 1.96945, -0.647519, 1.4122, 0.723005, -2.27766, -0.232736, -0.0591647, 0.058921]
const C_K = 1
const C_C = 1
const C_A = 0.692204
const C_B = -0.16909
const THETA0 = Float64[1.69361, 1.31758, 0.380231, 1.12343, 1.14859]

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
