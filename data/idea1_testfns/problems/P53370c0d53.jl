module P53370c0d53

const N = 10
const C_S = Float64[1.52194, 0.272308, 0.705853, 0.376655, -0.979403, 2.24286, 0.46196, -2.10102, -0.530603, -3.33304]
const C_V = Float64[0.270514, 0.432299, 0.609833, -1.20501, 0.0891542, -0.460313, -0.573725, -0.0950494, -2.02545, -1.01403, 1.2792, 0.347539, 2.03015, -1.34616, 2.05789, 0.219425, -1.43683, 0.00148226, -0.92825, 0.364111, 0.744761, 0.448313, -0.737218, -0.782464, 1.01629, -0.21341, -0.502532, 0.604606, -0.466723, -0.300877]
const C_D = Float64[2.78256, 464.159, 1, 1291.55, 3593.81, 59.9484, 166.81, 21.5443, 7.74264, 10000]
const C_W = 0.825461
const C_A = 3.98009
const C_B = -3.18438
const THETA0 = Float64[8.37819, -0.299318, -1.19685, 2.90231, -0.895264, 7.16912, 3.76948, -0.476896, 4.0301, -2.81615]

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
