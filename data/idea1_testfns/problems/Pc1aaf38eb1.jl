module Pc1aaf38eb1

const N = 2
const C_S = Float64[-0.220651, 2.17363]
const C_V = Float64[1.09464, -0.467098, -1.80325, 0.0480309, 2.70561, 1.36588]
const C_D = Float64[1, 1]
const C_A = 3.18572
const C_B = -1.25655
const THETA0 = Float64[-1.5893, 2.13933]

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
