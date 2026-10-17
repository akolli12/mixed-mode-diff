module P248efc1cf7

const N = 10
const C_S = Float64[-1.47789, 0.78264, -1.15193, 0.69712, -0.844326, 2.76739, -0.726138, -0.684842, 0.529325, 1.21106]
const C_V = Float64[-1.17959, -0.0331813, 0.413823, -0.16323, -0.471308, 0.109895, 1.84558, -1.1414, 0.70249, -0.293181, 2.25243, -0.110393, 0.110502, 0.577499, 1.78435, 1.38682, -0.0807631, -0.446877, -0.224107, 0.481183, -1.24985, -0.430417, 1.02509, -0.0206958, 0.791462, 2.00075, 0.898281, 0.179611, -2.25907, 0.212976]
const C_D = Float64[59.9484, 2.78256, 21.5443, 464.159, 3593.81, 166.81, 7.74264, 1, 10000, 1291.55]
const C_W = 0.727212
const C_A = 2.21852
const C_B = -0.853232
const THETA0 = Float64[-2.92668, 1.81397, 4.16462, -2.2589, -1.44634, 3.9287, 1.5644, -4.03213, 2.86893, -3.00922]

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
