module P6a7e06da45

const N = 5
const C_S = Float64[3.98687, -0.747089, -0.275979, -0.204417, 0.181116]
const C_V = Float64[1.50853, 1.19238, 1.07088, 1.61053, -0.181973, -1.57142, -1.05177, 1.62792, 0.626517, 0.985484, -0.535534, 1.32151, 0.387612, -0.00878746, -0.151017]
const C_K = 100
const C_C = 1
const C_A = 1.99044
const C_B = 1.79792
const THETA0 = Float64[-3.11103, 0.50783, 2.19371, -1.26586, 1.06071]

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
