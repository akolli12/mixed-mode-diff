module P3b9f72f24b

const N = 2
const C_S = Float64[1.78356, -1.55758]
const C_V = Float64[1.14376, 1.09868, 2.19366, 1.61315, -0.355227, -1.10195]
const C_D = Float64[1, 10]
const C_W = 0.870632
const C_A = 0.517639
const C_B = -0.831135
const THETA0 = Float64[5.3261, 3.71557]

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
