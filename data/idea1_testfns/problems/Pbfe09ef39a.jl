module Pbfe09ef39a

const N = 5
const C_S = Float64[-3.6087, -0.656462, 0.111424, -0.337471, 0.38939]
const C_V = Float64[-1.89769, -1.50542, -0.499792, -0.570717, -0.968866, -0.727669, 1.25467, -1.39585, 0.697256, -1.35649, 0.175955, -1.07743, 1.10508, -0.148407, 0.98281]
const C_K = 100
const C_C = 1
const C_A = 0.738695
const C_B = 1.56279
const THETA0 = Float64[1.62539, 2.28231, 1.422, -0.21048, 3.10779]

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
