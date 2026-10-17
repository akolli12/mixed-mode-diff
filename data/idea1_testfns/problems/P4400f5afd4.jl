module P4400f5afd4

const N = 2
const C_S = Float64[0.875867, 1.58611]
const C_V = Float64[0.0243844, -0.189061, 1.59938, -0.701485, 0.208019, -1.77922]
const C_D = Float64[100, 1]
const C_W = 0.683456
const C_A = 1.99494
const C_B = 0.826913
const THETA0 = Float64[-5.62216, -2.19428]

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
