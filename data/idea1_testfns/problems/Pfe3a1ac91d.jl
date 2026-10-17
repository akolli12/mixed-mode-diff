module Pfe3a1ac91d

const N = 2
const C_S = Float64[2.00122, 0.791508]
const C_V = Float64[1.6266, 1.47529, -0.220941, 0.0787943, -0.665706, 0.394934]
const C_D = Float64[100, 1]
const C_W = 0.791203
const C_A = 1.414
const C_B = 3.05759
const THETA0 = Float64[-2.01585, 2.82082]

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
        u = abs(z[i] / C_W)
        s += C_D[i] * C_W^2 * (u + log1p(exp(-2u)) - log(2.0))
    end
    return C_A * s + C_B
end

end # module
