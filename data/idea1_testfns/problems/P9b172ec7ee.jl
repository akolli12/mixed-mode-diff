module P9b172ec7ee

const N = 5
const C_S = Float64[-2.09555, -1.31387, -0.54001, -1.4838, 3.0229]
const C_V = Float64[-0.650056, -0.194981, -0.53972, -0.551021, 0.275431, -0.0271464, 1.88269, 0.558501, -0.0493381, -0.179294, -1.13673, 0.950488, -0.208997, -0.691391, 2.22527]
const C_D = Float64[1000, 10000, 100, 10, 1]
const C_W = 0.462871
const C_A = 0.73284
const C_B = 0.470985
const THETA0 = Float64[-6.23484, 2.19763, 0.891395, 1.90282, -4.48844]

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
