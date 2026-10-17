module P47ea4bd260

const N = 5
const C_S = Float64[0.531998, 1.69357, 2.10831, -0.355529, 0.152415]
const C_V = Float64[0.8164, 0.0719708, -0.502947, 0.201665, -1.07286, 0.579145, -1.02011, 1.80624, 0.986213, 0.343357, -1.08702, -0.000884428, -0.440194, -2.54317, -1.79692]
const C_D = Float64[1, 1.77828, 5.62341, 10, 3.16228]
const C_W = 0.67219
const C_A = 2.38052
const C_B = 2.4171
const THETA0 = Float64[2.61479, 0.492543, 0.552552, 2.15655, 4.44814]

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
