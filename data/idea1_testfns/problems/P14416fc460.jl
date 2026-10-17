module P14416fc460

const N = 5
const C_S = Float64[-0.235086, 2.468, -0.115133, 0.0440642, 1.59008]
const C_V = Float64[-1.26912, -1.82254, -0.556223, 0.394475, 0.906293, -0.646363, 0.379516, 0.221778, -1.24008, 0.0380465, 1.56487, 0.287727, -0.724105, -0.0870077, -0.33831]
const C_SC = Float64[1, 1, 1, 1, 1]
const C_FR = Float64[12.5267, 9.1358, 19.7629, 6.09963, 15.6299]
const C_PH = Float64[4.45336, 0.889112, 4.52051, 3.63442, 5.69908]
const C_Y = Float64[-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1]
const C_A = 3.60741
const C_B = -4.76062
const THETA0 = Float64[1.07301, 1.82655, -1.77413, 0.979093, 2.39234]

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
