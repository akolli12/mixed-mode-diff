module P061afc186a

const N = 2
const C_S = Float64[2.35694, 0.533521]
const C_V = Float64[0.373863, 0.505958, -0.235422, -0.416665, 0.163125, -0.579567]
const C_D = Float64[1, 10000]
const C_W = 0.821386
const C_A = 4.34263
const C_B = -1.54077
const THETA0 = Float64[5.26708, -2.93716]

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
