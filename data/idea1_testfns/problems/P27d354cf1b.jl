module P27d354cf1b

const N = 5
const C_S = Float64[1.20241, -0.228761, -2.4617, -1.73276, 2.46871]
const C_V = Float64[-0.0723268, -0.528045, -0.231266, 1.28118, -0.586855, 0.536571, -0.755367, -0.80064, -0.745694, 1.11593, 0.323149, -0.236522, -1.54854, -0.780434, -0.148333]
const C_SC = Float64[1, 1, 1, 1, 1]
const C_FR = Float64[25.9306, 24.8289, 18.5893, 2.37423, 4.23024]
const C_PH = Float64[2.25469, 3.90299, 4.0915, 1.71531, 1.91308]
const C_Y = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1]
const C_A = 0.752468
const C_B = -4.95302
const THETA0 = Float64[0.0804459, 1.24312, 1.48001, 1.02733, 2.63702]

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
