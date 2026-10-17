module P3bae916921

const N = 10
const C_S = Float64[0.861077, -0.871907, -0.0409837, 1.7856, -0.840755, 0.0539783, 2.22278, 0.772003, 1.7688, 0.993256]
const C_V = Float64[-0.047079, 0.249365, 1.08331, -1.41817, 0.257261, -0.556452, 1.9463, 1.51089, -0.207489, 0.0783608, -1.54242, -0.220636, 0.214494, 0.891912, 0.674779, -1.14057, -0.0906696, 2.15817, -0.84374, 0.336044, 0.875275, 0.791054, 0.231536, 1.66973, 0.323889, 1.30611, -1.13075, 0.23612, 1.2875, 0.283358]
const C_SC = Float64[1.6681, 59.9484, 2.78256, 21.5443, 35.9381, 7.74264, 100, 4.64159, 1, 12.9155]
const C_FR = Float64[23.2181, 12.887, 23.8327, 1.26015, 17.0088, 27.6113, 17.4465, 23.7185, 22.1031, 26.0807]
const C_PH = Float64[2.71402, 0.998674, 3.12943, 4.98207, 1.46217, 5.14481, 0.405806, 4.37017, 0.372766, 2.43138]
const C_Y = Float64[1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1]
const C_A = 0.594632
const C_B = 4.64374
const THETA0 = Float64[0.319725, -1.49088, -0.942013, 1.70173, -1.51056, 0.0476819, 1.23476, -1.45464, 1.62475, 0.73272]

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
