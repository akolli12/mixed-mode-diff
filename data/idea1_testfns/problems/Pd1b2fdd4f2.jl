module Pd1b2fdd4f2

const N = 10
const C_S = Float64[-2.62001, 1.75876, 0.530093, 1.47475, -0.978865, 1.51385, 2.7969, -0.627698, -0.587561, -0.145705]
const C_V = Float64[0.473977, 2.29558, 0.991746, 0.643941, -0.986395, -0.624396, -1.73495, -0.967359, 0.63835, 0.531928, 0.849052, 0.454356, -1.35195, -1.23757, 1.70379, 0.57856, -0.756175, -2.04974, 0.434143, 0.547323, -0.40995, -0.681252, 0.155686, 0.641873, -0.49665, -0.820102, -1.46242, -0.719528, 0.0795369, 0.860179]
const C_D = Float64[1, 2.15443, 100, 1000, 215.443, 46.4159, 4.64159, 464.159, 10, 21.5443]
const C_W = 0.370884
const C_A = 2.76704
const C_B = -2.75215
const THETA0 = Float64[-0.356517, -0.723555, -1.75997, 4.96382, 5.85961, -1.66271, 1.11166, -4.90384, 4.04813, -0.407524]

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
