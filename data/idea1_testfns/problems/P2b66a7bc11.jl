module P2b66a7bc11

const N = 10
const C_S = Float64[0.898049, -2.2706, -0.501732, -0.165457, -1.84491, 1.57034, -2.07046, 0.296866, -1.9396, -1.78407]
const C_V = Float64[-0.908169, -0.947976, -1.28969, 0.31414, 0.161579, 1.16071, -0.72093, -0.110393, -1.09991, -0.602044, 1.14309, -0.0823842, 0.806647, -1.89916, -0.951348, -0.468501, -0.128765, 0.574306, -0.150191, 1.45309, -0.850482, 1.10692, -0.591831, -0.814638, 0.0520466, 1.73001, -0.0309691, 0.714762, -2.00977, 1.15169]
const C_K = 100
const C_C = 1
const C_A = 1.23562
const C_B = -2.36754
const THETA0 = Float64[2.46692, -1.10675, 2.85517, -0.751627, -2.72153, -2.14982, -1.04864, -0.19867, 0.409663, -1.09697]

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
    @inbounds for i in 1:N-1
        s += C_K * (z[i+1] - z[i]^2)^2 + (C_C - z[i])^2
    end
    return C_A * s + C_B
end

end # module
