module P04de2310d6

const N = 5
const C_S = Float64[1.69117, -3.90578, 2.88214, 1.09681, 2.275]
const C_V = Float64[0.433538, -0.952491, 0.847767, 0.540111, -0.503413, -1.50559, -0.557568, 0.870515, 0.372675, 0.80351, -0.236932, -0.538695, 0.783948, -1.99642, -1.14587]
const C_D = Float64[100, 31.6228, 1, 10, 3.16228]
const C_W = 0.753589
const C_A = 0.888662
const C_B = -4.1184
const THETA0 = Float64[0.428572, 4.72226, -8.14325, -2.97683, 1.41852]

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
