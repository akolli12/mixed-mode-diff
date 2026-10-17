module Pbcd5276451

const N = 2
const C_S = Float64[2.53194, 3.14891]
const C_V = Float64[0.0530365, 0.885088, 0.58855, -0.585523, 0.986826, 1.23025]
const C_D = Float64[10, 1]
const C_A = 1.63649
const C_B = 1.4654
const THETA0 = Float64[-2.44456, 4.26235]

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
