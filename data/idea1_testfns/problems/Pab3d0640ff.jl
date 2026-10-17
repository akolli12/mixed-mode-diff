module Pab3d0640ff

const N = 2
const C_S = Float64[1.18729, 1.5717]
const C_V = Float64[-1.44005, -0.112923, 0.211371, 0.446132, 1.52734, 1.65764]
const C_D = Float64[10, 1]
const C_A = 4.3041
const C_B = -4.93304
const THETA0 = Float64[-0.544401, 0.524959]

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
