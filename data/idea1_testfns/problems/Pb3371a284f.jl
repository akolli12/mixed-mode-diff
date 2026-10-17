module Pb3371a284f

const N = 10
const C_S = Float64[0.546784, -1.77652, -1.11681, -0.167573, 2.56673, 0.633044, 4.27954, -2.56504, -0.857462, -1.76002]
const C_V = Float64[-0.0763822, 0.700505, 0.399928, 0.902814, 0.154014, -1.93754, -0.531778, -0.0117919, 0.859138, -1.49086, 1.23193, -1.44271, -1.69368, 0.434071, -0.439315, 0.471363, -1.04046, 0.302537, 0.705237, 0.880411, 1.18013, 2.08524, -2.22737, -1.61218, 0.037086, 0.679694, 1.06436, 0.489813, 1.38646, -0.00263599]
const C_D = Float64[1291.55, 10000, 59.9484, 464.159, 7.74264, 2.78256, 21.5443, 166.81, 3593.81, 1]
const C_A = 0.521822
const C_B = 0.3002
const THETA0 = Float64[1.08606, -4.25554, -2.30315, 2.3034, 2.91605, -1.4127, 0.935121, -3.84895, 0.443607, -3.76403]

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
    s = zero(T)
    @inbounds for i in 1:N
        s += C_D[i] * z[i]^2
    end
    return C_A * 0.5 * s + C_B
end

end # module
