module P7d449298cd

const N = 5
const C_S = Float64[0.218682, 0.4677, -2.185, 0.935611, 0.450282]
const C_V = Float64[-1.57189, -0.0320074, -0.923604, -0.0400165, 1.39925, 0.279842, 1.68671, 2.03257, -0.37972, 0.765994, -1.46354, 2.14476, 0.217079, -0.266146, 0.173496]
const C_K = 1
const C_C = 1
const C_A = 1.10689
const C_B = 4.57528
const THETA0 = Float64[1.09492, 2.59625, 0.625565, -0.283222, 0.100621]

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
