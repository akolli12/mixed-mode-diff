module Pd6b2497d55

const N = 2
const C_S = Float64[-1.9943, 0.211776]
const C_V = Float64[-1.02595, 1.19448, 0.253491, 0.365412, 0.720163, 1.63806]
const C_D = Float64[1, 1]
const C_A = 2.28992
const C_B = 1.79415
const THETA0 = Float64[0.40577, -3.28141]

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
