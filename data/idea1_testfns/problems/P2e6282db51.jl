module P2e6282db51

const N = 10
const C_S = Float64[0.627669, 2.44159, 1.74395, -1.06219, -1.93212, -1.54876, -0.935915, 1.68078, -0.287453, 0.72806]
const C_V = Float64[0.559971, -0.586006, 1.01535, 0.118866, 0.143208, 0.388402, -0.482981, 0.801036, -1.13585, -0.438447, -0.47239, 2.6054, -0.108071, 1.22484, 0.714156, 0.0321029, -0.417713, -0.0455283, -1.77217, 0.558241, -0.0162447, 0.0957265, 0.116416, 0.63407, 0.113419, 0.969838, -0.00665641, -0.897641, 2.65099, 1.47187]
const C_D = Float64[1.6681, 2.78256, 4.64159, 3.59381, 1.29155, 5.99484, 1, 7.74264, 2.15443, 10]
const C_A = 2.65129
const C_B = -0.669836
const THETA0 = Float64[2.13782, -0.468174, 2.02015, -0.710632, -2.71404, -0.971042, -0.911882, 2.32111, -0.667686, 1.25019]

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
