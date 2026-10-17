module P38e3d1ac1c

const N = 2
const C_S = Float64[-0.167304, -0.0778867]
const C_V = Float64[0.174024, -0.24493, -1.2447, 0.960632, 0.819781, 0.453153]
const C_D = Float64[10000, 1]
const C_A = 1.70207
const C_B = -1.42538
const THETA0 = Float64[1.46063, -0.612733]

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
