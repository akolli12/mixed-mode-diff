module P4e7af1ce81

const N = 2
const C_S = Float64[0.430096, -0.728198]
const C_V = Float64[1.42699, 0.331903, 1.3036, -1.77214, 0.121003, 0.228104]
const C_D = Float64[1, 1]
const C_W = 0.609226
const C_A = 3.67895
const C_B = 0.199222
const THETA0 = Float64[-2.1791, 4.52162]

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
