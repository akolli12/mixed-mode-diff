module P22a7a3a38f

const N = 10
const C_S = Float64[-2.10999, 0.00636584, -0.0596959, 0.628969, -3.36716, -1.89994, 1.88294, -0.681887, -1.88624, 1.67628]
const C_V = Float64[-0.213542, -0.425831, 0.966235, 0.188348, 0.26585, 0.358579, 0.961382, 0.635528, -0.472594, 1.37182, -0.365825, 1.73334, -0.311585, -2.24571, 1.02635, -0.307063, -0.536616, 0.845115, -1.53509, -0.615606, 0.447472, 0.169768, -0.070489, -0.342183, -1.11621, -1.35028, -1.50722, -2.23094, 0.217017, 0.587124]
const C_SC = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const C_FR = Float64[1.86098, 19.2171, 15.1489, 10.8731, 25.9856, 25.8349, 25.7872, 27.0214, 11.4489, 24.0046]
const C_PH = Float64[2.24017, 2.42289, 3.59868, 3.05297, 3.07474, 4.88719, 2.73466, 0.548425, 3.87104, 4.72305]
const C_Y = Float64[-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
const C_A = 2.76903
const C_B = -3.86276
const THETA0 = Float64[-0.978757, 0.805753, -1.87958, 0.689499, -0.77409, -1.78698, -0.371564, -0.727517, -2.66468, -2.77401]

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
