module P52b5f83562

const N = 5
const C_S = Float64[-0.047515, 4.07116, -0.175789, 1.24909, -3.3257]
const C_V = Float64[0.662634, 1.1598, 0.610101, -0.336723, -1.7879, 0.139538, 1.1733, 0.0173955, -0.548758, 2.23064, -0.291065, -0.441794, -0.0548884, 1.06652, -1.53384]
const C_D = Float64[1, 100, 10, 31.6228, 3.16228]
const C_A = 1.10973
const C_B = 3.15391
const THETA0 = Float64[-2.47215, -4.36309, -1.65906, 0.897266, 3.46172]

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
