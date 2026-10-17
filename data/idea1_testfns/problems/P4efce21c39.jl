module P4efce21c39

const N = 10
const C_S = Float64[1.70259, 1.88171, 0.674377, 1.87154, -0.73463, 0.42965, -1.27065, -2.07014, 0.971074, -0.139714]
const C_V = Float64[0.208473, -1.29665, 1.14708, 1.89011, 0.612901, 0.490029, 2.20036, -0.655376, 1.08876, 1.57995, -0.362826, 0.0622674, 0.256349, 0.144272, 1.17126, 0.0611374, -0.777245, 1.35505, 1.36492, 1.60394, -0.369941, -1.16352, -0.375736, -2.6237, 0.362869, -1.72011, 0.718652, 0.125352, -1.48763, 0.159384]
const C_SC = Float64[35.9381, 4.64159, 12.9155, 59.9484, 2.78256, 21.5443, 7.74264, 1, 100, 1.6681]
const C_FR = Float64[11.3302, 26.051, 21.7886, 16.4111, 9.40965, 12.3455, 18.5809, 16.4985, 4.31695, 17.9428]
const C_PH = Float64[4.40052, 5.01786, 0.665517, 4.75383, 2.637, 2.74882, 6.17869, 3.41937, 4.97503, 2.49638]
const C_Y = Float64[1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
const C_A = 0.547388
const C_B = -3.55142
const THETA0 = Float64[0.543048, -0.223417, 0.858198, -1.60026, 2.57028, -2.21245, -0.863202, -0.538982, 1.3016, 2.5522]

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
