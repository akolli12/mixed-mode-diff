module P372706f407

const N = 2
const C_S = Float64[1.05702, 1.03698]
const C_V = Float64[-1.41266, -0.383882, 1.17156, -0.610624, 0.0094425, 0.0424665]
const C_K = 1
const C_C = 1
const C_A = 3.36063
const C_B = 1.3677
const THETA0 = Float64[0.417641, 0.411152]

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
