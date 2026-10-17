module P15942f25c8

const N = 5
const C_S = Float64[-2.94025, -2.0983, 1.84732, 0.0782003, 0.0226846]
const C_V = Float64[1.33404, -0.738182, 1.55121, 0.148618, 0.42724, 0.203937, 0.390283, 0.684529, -1.25076, -0.434758, 0.333459, -0.555428, -0.997077, 0.999669, -0.727381]
const C_K = 10
const C_C = 1
const C_A = 0.841293
const C_B = 4.85854
const THETA0 = Float64[-2.71039, -3.10929, 2.12379, -0.486352, -1.90095]

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
