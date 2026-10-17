module Pbfbde1dc51

const N = 5
const C_S = Float64[-1.23633, -2.25527, -0.651532, -1.93964, -0.733562]
const C_V = Float64[0.6518, -1.30412, 1.94254, -1.08663, 0.356842, -1.22374, -0.538133, -1.92401, 0.343439, -0.472474, -0.324554, -0.883504, -0.0509061, 0.886476, -0.85501]
const C_SC = Float64[1, 10, 31.6228, 100, 3.16228]
const C_FR = Float64[26.9314, 29.7655, 1.21998, 18.8552, 2.75754]
const C_PH = Float64[4.85376, 2.14021, 5.32984, 2.70384, 0.143003]
const C_Y = Float64[1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1]
const C_A = 4.40943
const C_B = -4.00954
const THETA0 = Float64[0.240922, -0.0694242, -2.96635, -1.07721, -0.694459]

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
