module P3a838851bd

const N = 2
const C_S = Float64[-2.56987, 1.55541]
const C_V = Float64[1.53218, -0.40936, -0.132431, -0.659696, 1.74313, 1.70984]
const C_K = 100
const C_C = 1
const C_A = 0.875554
const C_B = -0.557015
const THETA0 = Float64[0.810459, -3.00925]

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
