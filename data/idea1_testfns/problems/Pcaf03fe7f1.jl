module Pcaf03fe7f1

const N = 5
const C_S = Float64[0.297463, 0.855507, 1.62703, 1.58318, 0.199682]
const C_V = Float64[-1.74614, 1.86356, -0.283243, 1.08581, -1.25228, -0.541357, 1.07722, 0.435743, -0.764484, 2.38453, 0.239339, 0.689015, -0.44515, 0.876087, 1.66558]
const C_SC = Float64[31.6228, 3.16228, 100, 10, 1]
const C_FR = Float64[17.3894, 7.59223, 3.54388, 17.6327, 11.5164]
const C_PH = Float64[2.98049, 2.66279, 5.52765, 2.19863, 0.553954]
const C_Y = Float64[-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
const C_A = 3.5044
const C_B = -4.1684
const THETA0 = Float64[-0.171647, 0.26215, 2.2488, 0.283585, 0.373514]

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
