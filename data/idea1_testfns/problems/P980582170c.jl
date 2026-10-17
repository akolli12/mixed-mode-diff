module P980582170c

const N = 5
const C_S = Float64[1.67363, -1.19196, -0.133137, -2.47867, 0.698716]
const C_V = Float64[-1.91191, 1.34419, 0.0410531, -0.561639, 0.575346, 1.01781, 0.629764, 1.22925, -2.42245, 0.701232, 0.456205, 1.01201, -0.596629, -0.160253, 1.73851]
const C_SC = Float64[10, 3.16228, 5.62341, 1, 1.77828]
const C_FR = Float64[24.5778, 15.9685, 8.82889, 15.9867, 5.00882]
const C_PH = Float64[1.64144, 3.84391, 2.89959, 4.85862, 0.657502]
const C_Y = Float64[1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1]
const C_A = 1.01545
const C_B = 4.5075
const THETA0 = Float64[-3.19399, -0.500205, -0.948063, -0.386185, -0.54551]

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
