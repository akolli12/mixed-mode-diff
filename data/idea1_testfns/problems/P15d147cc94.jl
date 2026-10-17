module P15d147cc94

const N = 5
const C_S = Float64[3.53069, 0.709702, 0.422928, -0.520776, -0.0553472]
const C_V = Float64[-0.389122, 1.81667, -0.222551, 0.183071, 0.581142, 1.00662, 0.0547426, -0.72394, -0.417902, 1.35927, -0.385527, -0.0645112, -1.07967, 0.999275, -0.359011]
const C_SC = Float64[3.16228, 1, 1.77828, 5.62341, 10]
const C_FR = Float64[22.385, 4.73823, 21.8744, 28.4869, 18.4596]
const C_PH = Float64[3.44118, 5.2664, 4.14009, 4.8429, 4.86096]
const C_Y = Float64[1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1]
const C_A = 4.77239
const C_B = -0.566631
const THETA0 = Float64[1.14706, 1.82761, -0.880233, 2.23055, -1.46824]

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
