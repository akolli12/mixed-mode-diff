module P72dd556487

const N = 5
const C_S = Float64[-1.4954, 1.58333, -1.79877, 2.33226, -0.568474]
const C_V = Float64[0.387584, -0.602934, -0.905518, 0.8676, 0.52683, -0.186302, -0.553245, -2.34399, -1.29798, -0.231073, -0.143488, -1.11226, 0.687903, -0.664141, 1.56952]
const C_D = Float64[1, 1.77828, 3.16228, 10, 5.62341]
const C_A = 1.47781
const C_B = 0.322975
const THETA0 = Float64[-3.7526, 1.31053, 1.20769, -2.33939, 2.90525]

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
