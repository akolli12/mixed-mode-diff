module Pf3a9c248a3

const N = 10
const C_S = Float64[-1.75207, -1.09264, -1.865, 1.97149, 1.51953, 1.93763, -0.229357, -2.40827, 2.4021, 0.280761]
const C_V = Float64[1.30442, -1.81201, 0.421958, -0.692772, -0.974269, -0.0133762, 0.7196, 1.37782, -0.660971, 1.65613, -1.27756, 1.28277, 0.025627, 3.60133, -1.50017, 0.747034, 1.11746, -0.814358, 1.27709, -0.182316, 1.5055, 2.25199, 0.0452842, -0.253208, 0.647101, 0.0304007, -0.196121, -0.93186, -0.104437, 0.543851]
const C_D = Float64[2.78256, 464.159, 166.81, 10000, 1291.55, 7.74264, 59.9484, 1, 3593.81, 21.5443]
const C_W = 0.477395
const C_A = 1.23747
const C_B = 1.75828
const THETA0 = Float64[-0.0265785, -6.90314, 1.78354, -2.29861, -1.23257, 5.47377, -1.81121, -4.80219, -0.690986, -0.672447]

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
