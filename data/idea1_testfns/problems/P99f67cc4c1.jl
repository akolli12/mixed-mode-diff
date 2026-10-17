module P99f67cc4c1

const N = 5
const C_S = Float64[1.18742, 1.8851, -0.128119, 3.33458, 2.26828]
const C_V = Float64[-1.19014, 0.684431, 0.961808, 0.22274, -0.138039, -1.90569, 0.722359, 0.138435, -0.0310333, -0.786373, -0.304617, 1.48208, 0.210396, -0.346759, -1.08659]
const C_D = Float64[10, 1, 31.6228, 3.16228, 100]
const C_A = 2.58964
const C_B = 2.61185
const THETA0 = Float64[2.66001, 1.05672, 0.298137, 2.70458, 1.71432]

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
