module P2828266078

const N = 2
const C_S = Float64[-0.268738, 2.74464]
const C_V = Float64[2.35397, 0.379127, 0.28548, -0.780775, -1.79278, 1.12184]
const C_D = Float64[1, 1]
const C_W = 0.641247
const C_A = 0.526336
const C_B = -1.82287
const THETA0 = Float64[-3.48753, -4.10537]

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
        u = abs(z[i] / C_W)
        s += C_D[i] * C_W^2 * (u + log1p(exp(-2u)) - log(2.0))
    end
    return C_A * s + C_B
end

end # module
