module P1fc4840fe2

const N = 10
const C_S = Float64[-1.64788, 1.30347, 2.15606, -3.7805, 0.479871, -1.15401, 1.97197, -2.14427, -0.678545, -1.24791]
const C_V = Float64[-0.973378, 0.00289739, -0.491914, -1.1668, 1.4941, 0.576833, -2.07774, -0.404697, -1.76062, 0.501418, 0.853904, 0.410517, 0.925162, -0.773848, -0.462675, 0.885367, -1.10806, -2.18416, -0.228043, -0.277009, 0.37, 0.636106, 0.987559, -0.305816, 1.30891, -1.79551, 0.395804, -1.63927, 0.179132, 0.744702]
const C_K = 10
const C_C = 1
const C_A = 1.54492
const C_B = -0.0744643
const THETA0 = Float64[-3.0217, -0.929955, -0.857995, -2.85462, -3.27749, 0.965039, 1.67134, 1.43173, -0.521058, -3.00966]

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
    @inbounds for i in 1:N-1
        s += C_K * (z[i+1] - z[i]^2)^2 + (C_C - z[i])^2
    end
    return C_A * s + C_B
end

end # module
