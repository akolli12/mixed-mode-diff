module Pff2e933d9d

const N = 5
const C_S = Float64[-2.78715, 2.14285, 0.103167, -0.683514, 0.575697]
const C_V = Float64[1.80356, 1.45519, 0.573984, -1.07812, -0.914667, 1.34734, -0.072477, -0.274679, 0.287292, -1.51919, -1.4891, -0.614898, -2.40544, -0.688325, -0.992514]
const C_SC = Float64[1, 1, 1, 1, 1]
const C_FR = Float64[3.2273, 20.8283, 2.62363, 12.4104, 23.989]
const C_PH = Float64[1.87292, 3.12195, 1.97716, 1.20384, 5.38396]
const C_Y = Float64[1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
const C_A = 0.581146
const C_B = 0.0823808
const THETA0 = Float64[-3.14768, 1.07688, -0.756164, 3.33473, 0.274053]

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
