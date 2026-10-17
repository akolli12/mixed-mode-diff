module P1cc254e6b8

const N = 2
const C_S = Float64[-2.17827, -1.72611]
const C_V = Float64[-0.535657, 0.603822, 0.264167, 0.0947816, 2.78297, 0.351092]
const C_D = Float64[1, 10000]
const C_A = 1.65227
const C_B = -2.61107
const THETA0 = Float64[-1.50715, -1.8115]

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
