module P500f8d3dd6

const N = 10
const C_S = Float64[-1.73018, -0.0200893, -4.24986, -0.3838, 3.49083, -0.555829, -2.18627, -0.0542937, -1.24908, -0.519595]
const C_V = Float64[-1.37613, -0.246271, 0.745471, 0.711738, -0.633697, 1.32307, -0.886137, -0.95747, 1.58711, 0.410501, -0.573332, 1.10074, -0.538179, -0.920296, 0.345535, 2.31692, -0.217694, 1.17338, -1.6879, -1.56624, -1.30759, 0.369294, -0.355273, -0.983388, 0.241391, -0.211242, -0.659254, -0.0425701, -0.8338, 1.32841]
const C_K = 10
const C_C = 1
const C_A = 4.98339
const C_B = -3.90773
const THETA0 = Float64[0.372119, -2.51846, -2.53225, 3.18082, 1.64589, -0.527996, -1.51146, -1.81339, 2.67314, -3.8286]

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
