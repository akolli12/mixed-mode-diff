module P8a46e146e1

const N = 5
const C_S = Float64[-1.6956, -0.812058, -0.554408, 2.15845, -0.653286]
const C_V = Float64[-0.504243, -1.38531, -0.890984, 0.0316596, 1.06699, -1.31405, 0.668482, -1.43323, -1.54702, -0.233151, 0.77897, 1.99609, 0.623733, 0.985345, 0.398178]
const C_D = Float64[1, 1, 1, 1, 1]
const C_A = 0.936817
const C_B = 1.27282
const THETA0 = Float64[-0.270241, 1.27598, 0.00619106, 0.453655, -1.24725]

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
        s += C_D[i] * z[i]^2
    end
    return C_A * 0.5 * s + C_B
end

end # module
