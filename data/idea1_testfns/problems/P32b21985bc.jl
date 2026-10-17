module P32b21985bc

const N = 5
const C_S = Float64[0.376332, 0.985035, 2.00426, 2.03233, -0.685572]
const C_V = Float64[1.10782, -1.36781, 0.368818, 1.40975, 1.04472, -0.30757, -1.55617, -1.22296, -0.151005, -0.103205, -0.575714, 0.914607, 1.11446, 0.540087, 1.4496]
const C_D = Float64[31.6228, 177.828, 1, 5.62341, 1000]
const C_W = 0.497779
const C_A = 2.6941
const C_B = 1.07696
const THETA0 = Float64[6.04595, 0.500156, 2.88825, -1.62308, -3.37118]

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
