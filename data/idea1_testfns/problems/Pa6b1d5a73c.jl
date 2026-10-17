module Pa6b1d5a73c

const N = 2
const C_S = Float64[1.5668, 2.45383]
const C_V = Float64[-0.427494, -1.19348, -1.39067, -0.597922, -0.248136, -1.50592]
const C_D = Float64[1, 100]
const C_W = 0.845273
const C_A = 3.3413
const C_B = 1.053
const THETA0 = Float64[6.76428, 2.32639]

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
