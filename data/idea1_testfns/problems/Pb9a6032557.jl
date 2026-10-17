module Pb9a6032557

const N = 5
const C_S = Float64[3.10604, -0.876221, -1.33781, -2.53124, 2.32142]
const C_V = Float64[1.55868, -0.102446, -0.0894462, -1.40941, -2.71942, 1.30965, -0.136155, 0.515337, -0.45024, -0.74861, -1.50745, 0.437631, -1.18664, -0.453142, -0.139242]
const C_K = 10
const C_C = 1
const C_A = 3.56219
const C_B = 3.11063
const THETA0 = Float64[1.45086, -1.22613, -3.06208, -3.11895, 2.11925]

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
