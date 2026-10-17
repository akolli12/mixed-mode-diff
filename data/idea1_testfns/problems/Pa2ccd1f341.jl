module Pa2ccd1f341

const N = 5
const C_S = Float64[1.53633, 1.28529, 1.37694, 0.2376, 3.49893]
const C_V = Float64[0.237295, 0.031829, 0.427984, -0.214294, 2.0523, 3.09818, -0.973773, -0.912289, 0.416126, -0.0449787, 0.544549, -0.839507, 1.85465, 1.31462, 0.462714]
const C_D = Float64[10000, 1000, 100, 10, 1]
const C_A = 2.18318
const C_B = 1.76444
const THETA0 = Float64[-1.15309, 3.29188, -3.77297, -1.5509, -2.40689]

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
