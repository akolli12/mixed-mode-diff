module Pcc0fb2583e

const N = 10
const C_S = Float64[0.0339956, -0.249816, -2.99664, 1.87851, -1.55293, -1.33961, -2.48902, -1.99017, 0.651409, -0.599583]
const C_V = Float64[-1.95021, -1.09765, -0.651431, 0.572737, -0.372052, 0.635347, -0.576794, -0.130539, 0.399952, -1.97395, 0.313746, 0.0485218, -0.943234, -0.129153, -1.16147, -0.296312, -0.97775, -0.0304895, -1.64673, 0.110694, -0.362774, 1.03084, -0.118058, 0.304675, -1.01616, -0.579874, -0.795333, 0.293869, -0.346555, -0.357607]
const C_D = Float64[1, 2.15443, 46.4159, 215.443, 1000, 10, 100, 4.64159, 464.159, 21.5443]
const C_W = 0.904108
const C_A = 2.42323
const C_B = -3.54072
const THETA0 = Float64[4.53868, -1.67816, -1.3434, 4.18303, -2.78676, 5.97762, -1.25269, -5.32182, -1.6844, 2.35357]

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
