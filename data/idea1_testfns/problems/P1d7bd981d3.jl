module P1d7bd981d3

const N = 5
const C_S = Float64[-0.546646, -2.33559, -1.97544, 3.15661, -0.3076]
const C_V = Float64[-1.20227, -0.47758, -1.2421, 1.30448, -0.603476, -0.0854133, -0.542023, 0.715119, -0.376969, 1.12956, 1.68895, 0.331908, -0.363376, -0.231923, -0.35617]
const C_D = Float64[10000, 100, 1000, 10, 1]
const C_A = 0.750579
const C_B = 1.75773
const THETA0 = Float64[2.43759, -2.6636, 2.30754, -0.664759, 1.85301]

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
