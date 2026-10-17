module P11e3777dc6

const N = 2
const C_S = Float64[1.84342, -3.00285]
const C_V = Float64[1.66633, -0.547134, 0.889147, 0.842513, 1.33948, 1.90948]
const C_K = 10
const C_C = 1
const C_A = 2.37429
const C_B = -0.46074
const THETA0 = Float64[-2.59778, -2.84977]

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
