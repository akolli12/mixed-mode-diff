module P43aa2c0481

const N = 2
const C_S = Float64[-1.14268, -0.848579]
const C_V = Float64[-0.340458, -0.928379, 0.70278, 0.166631, -1.66009, -0.888239]
const C_K = 10
const C_C = 1
const C_A = 2.63534
const C_B = 3.25291
const THETA0 = Float64[-1.41805, 1.77702]

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
