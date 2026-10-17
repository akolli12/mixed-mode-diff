module P05cca2cbf4

const N = 10
const C_S = Float64[-1.66428, -1.10068, 0.288968, 2.25409, -2.35686, -1.57887, 2.75827, -1.30806, -0.29203, -3.18709]
const C_V = Float64[1.10854, 0.955078, 1.39276, -0.380366, 0.810529, 0.616315, -1.43301, 0.577537, -0.0754636, -0.63374, -0.217053, 1.09535, -1.42234, 0.30918, -0.177171, 1.41907, -1.27171, 0.605103, -0.277491, 1.27023, -1.0095, -0.755096, 0.124825, 0.0748475, 0.507196, -0.841489, -0.638871, -2.65789, -1.56999, -0.205847]
const C_SC = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const C_FR = Float64[7.81645, 24.9132, 2.35194, 4.81072, 13.0346, 3.46407, 10.7834, 11.8414, 14.8618, 17.7605]
const C_PH = Float64[2.86718, 0.43264, 5.7952, 2.78797, 5.34964, 5.53962, 4.55414, 1.78834, 0.153862, 4.12107]
const C_Y = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
const C_A = 0.685753
const C_B = -0.778619
const THETA0 = Float64[-0.350531, 3.83716, -1.72725, 3.17029, -0.3264, 2.68582, -1.70991, 3.06547, 1.44892, -2.87523]

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
