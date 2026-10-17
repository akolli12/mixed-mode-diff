module Pd5bba1f55f

const N = 2
const C_S = Float64[0.363711, -2.38184]
const C_V = Float64[-0.0158715, 0.762343, 0.896423, -0.0102789, 0.206798, 1.32375]
const C_D = Float64[100, 1]
const C_A = 1.84838
const C_B = 0.0199933
const THETA0 = Float64[-1.39769, -3.55728]

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
