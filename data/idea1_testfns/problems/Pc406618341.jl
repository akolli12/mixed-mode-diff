module Pc406618341

const N = 2
const C_S = Float64[-0.0967811, -2.55175]
const C_V = Float64[-1.35834, -0.388985, 0.909224, -1.83578, 0.262051, -1.30407]
const C_D = Float64[1, 1]
const C_A = 0.59088
const C_B = -2.78375
const THETA0 = Float64[1.51161, -2.05622]

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
