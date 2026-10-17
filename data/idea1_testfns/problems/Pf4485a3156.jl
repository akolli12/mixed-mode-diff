module Pf4485a3156

const N = 2
const C_S = Float64[-0.829176, 1.8593]
const C_V = Float64[-0.779833, 0.3945, -0.191656, -1.05204, -1.72066, -0.485005]
const C_D = Float64[1, 1000]
const C_A = 1.27536
const C_B = 1.44393
const THETA0 = Float64[0.254911, -0.990869]

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
