module Pcfb1427148

const N = 2
const C_S = Float64[2.51108, 0.500579]
const C_V = Float64[-0.667143, -1.34321, 0.282587, 2.80824, 0.92753, -1.90596]
const C_D = Float64[100, 1]
const C_A = 2.01716
const C_B = 0.91615
const THETA0 = Float64[1.86059, 1.19738]

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
