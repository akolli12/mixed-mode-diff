module P8a46682c8c

const N = 10
const C_S = Float64[0.574978, -0.0800334, 2.68734, 2.14403, 0.0757214, -2.58876, 0.983706, -1.64663, 2.73818, 0.609887]
const C_V = Float64[-0.531145, 0.753535, 0.987594, 0.0482012, -0.247659, -0.261857, -1.65077, -0.0371766, -0.0180521, -1.15117, -0.227162, 0.925361, -0.706908, -0.263689, 0.0666727, -0.654496, 0.552237, -0.812716, -0.331885, -0.717928, 1.49302, 1.739, -0.00640485, -0.552001, 0.654535, 0.247656, -1.18041, -2.31085, -0.301797, 0.0818855]
const C_D = Float64[7.74264, 12.9155, 35.9381, 21.5443, 1, 1.6681, 100, 4.64159, 59.9484, 2.78256]
const C_A = 0.600943
const C_B = 1.29765
const THETA0 = Float64[0.622444, 0.133353, 1.69435, 2.69193, -0.04887, -3.72301, -0.543808, -1.96935, 0.757937, 0.632899]

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
