module Pe3f912581b

const N = 10
const C_S = Float64[2.07435, -2.58915, -1.04441, 0.123758, -0.433352, 2.53924, 0.76752, -2.05771, 0.751973, -1.62123]
const C_V = Float64[-1.41325, -1.03118, -0.831571, -0.541526, -0.447792, 0.0979106, 2.55702, 0.995294, -0.926901, -0.478199, -0.721465, 1.46639, 0.209503, -1.16904, -1.72508, 0.197898, -0.0292495, -0.917466, -0.273886, 0.653786, 0.472053, 0.58917, 0.0421005, 2.09086, 0.186313, 0.522576, 0.115631, -0.338035, 0.646547, -0.11699]
const C_K = 10
const C_C = 1
const C_A = 1.44731
const C_B = 3.88592
const THETA0 = Float64[1.89884, -1.75153, -0.786935, -0.801778, -1.2504, 2.14167, -0.482852, -3.90555, -0.0083555, -2.02905]

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
