module P07ea8305c2

const N = 10
const C_S = Float64[-3.82211, 2.89325, 0.8259, -0.165079, -1.14409, -0.110025, 2.68942, -0.256756, -0.0923932, 1.57263]
const C_V = Float64[-0.0321937, 0.694639, -0.433682, 1.20848, 1.07725, 0.996249, 0.477193, -1.02732, -0.545949, -0.665356, 0.472994, 0.542763, 1.60239, -1.08372, -0.11976, 0.535894, -1.09812, 0.844914, -0.130878, 2.72193, 1.05828, 0.22921, 1.0695, -0.77405, 0.929385, 1.84679, 0.115308, 1.19357, -1.09069, -2.9886]
const C_D = Float64[1.6681, 3.59381, 5.99484, 1.29155, 2.78256, 7.74264, 4.64159, 10, 2.15443, 1]
const C_A = 2.08451
const C_B = -2.07268
const THETA0 = Float64[-3.08425, 1.58955, 3.44254, -2.42608, -0.563285, 2.0806, 2.26573, 2.57535, -2.2487, 0.699531]

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
