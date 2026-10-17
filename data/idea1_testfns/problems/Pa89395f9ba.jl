module Pa89395f9ba

const N = 5
const C_S = Float64[-1.52581, 1.20029, -1.46444, 0.28467, -3.23686]
const C_V = Float64[0.734541, 1.11574, 0.251607, -0.694989, -0.0899984, 0.0958333, -0.0687652, -0.237083, -0.604895, -0.488991, 0.24663, -1.17361, 0.204835, -1.00138, 2.23034]
const C_SC = Float64[3.16228, 1, 31.6228, 10, 100]
const C_FR = Float64[12.5964, 29.7243, 25.9446, 18.2617, 16.4301]
const C_PH = Float64[5.78795, 2.12673, 5.85192, 5.99711, 3.5325]
const C_Y = Float64[-1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
const C_A = 3.69836
const C_B = -4.64464
const THETA0 = Float64[0.741079, 1.11677, 0.272376, -3.08801, 2.29572]

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
