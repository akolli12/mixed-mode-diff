module P2b1c195456

const N = 5
const C_S = Float64[-0.326193, 2.00041, 0.782541, -1.17696, -0.401736]
const C_V = Float64[0.311233, -0.315174, 0.108472, 1.39557, 0.07211, 0.658519, -0.910742, 2.13177, -0.610557, 0.0344354, -1.98772, 0.107321, -1.44259, 0.70878, -0.00231709]
const C_D = Float64[10, 1000, 10000, 1, 100]
const C_W = 0.950314
const C_A = 4.36594
const C_B = 4.21369
const THETA0 = Float64[-2.00004, -0.325145, 6.67156, -1.47698, -1.43156]

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
