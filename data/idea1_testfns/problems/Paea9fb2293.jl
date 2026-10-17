module Paea9fb2293

const N = 5
const C_S = Float64[-2.68836, -1.11398, 1.70842, -3.24817, 1.06143]
const C_V = Float64[-0.892505, -0.227823, -0.531152, -1.00027, 0.22656, 0.865693, -1.04352, 0.61777, 0.737339, -0.800351, -0.496853, 0.9249, 0.71315, -0.288478, 0.665119]
const C_SC = Float64[3.16228, 5.62341, 10, 1.77828, 1]
const C_FR = Float64[15.4981, 18.83, 9.87759, 17.7182, 22.4628]
const C_PH = Float64[4.2109, 5.20743, 4.1192, 1.51174, 2.61239]
const C_Y = Float64[-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
const C_A = 0.515146
const C_B = 4.63816
const THETA0 = Float64[3.47571, -2.04222, 1.35455, 2.91127, -0.882406]

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
    w = z
    M = length(C_Y)
    s = zero(T)
    @inbounds for i in 1:M
        t = i / M
        dot = zero(T)
        for j in 1:N
            dot += C_SC[j] * cos(C_FR[j] * t + C_PH[j]) * w[j]
        end
        u = -C_Y[i] * dot
        s += u > 0 ? u + log1p(exp(-u)) : log1p(exp(u))
    end
    r = zero(T)
    @inbounds for j in 1:N
        r += w[j]^2
    end
    return C_A * (s / M + 0.5 * 0.01 * r) + C_B
end

end # module
