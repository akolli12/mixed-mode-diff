module P01602223e5

const N = 5
const C_S = Float64[1.28953, 1.88152, 3.18189, -0.267018, 0.444904]
const C_V = Float64[0.957552, 1.91778, -0.248765, 0.444693, 0.0746597, -0.312677, 0.0619117, 0.234013, -1.54929, -1.11795, 0.290234, 0.142861, 0.337621, 0.157194, 0.716104]
const C_K = 100
const C_C = 1
const C_A = 1.56561
const C_B = -1.83921
const THETA0 = Float64[-0.778547, -1.56583, 0.989868, 2.67328, 0.455751]

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
