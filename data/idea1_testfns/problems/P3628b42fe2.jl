module P3628b42fe2

const N = 2
const C_S = Float64[-0.339834, 0.314391]
const C_V = Float64[1.11586, 0.93874, 0.958901, -0.862257, -1.16201, -1.47265]
const C_D = Float64[1, 1000]
const C_W = 0.914638
const C_A = 1.99933
const C_B = 0.527184
const THETA0 = Float64[-0.264699, -4.5581]

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
