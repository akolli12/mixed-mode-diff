module Pc743481733

const N = 2
const C_S = Float64[-0.598514, -2.7653]
const C_V = Float64[-1.40041, 0.0869438, 1.17798, 1.57438, 0.306626, -1.25481]
const C_K = 1
const C_C = 1
const C_A = 4.6999
const C_B = -2.71665
const THETA0 = Float64[2.97547, 1.40125]

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
