module P956e30bc49

const N = 5
const C_S = Float64[-1.36891, 1.03199, -0.53846, -2.95113, -0.297508]
const C_V = Float64[2.55907, 1.12961, -0.106915, -1.66461, 1.48524, 1.34974, 1.43068, -1.8856, -0.521728, -0.829913, 0.296692, -1.83679, -2.23439, -0.281877, -0.553619]
const C_D = Float64[1, 1, 1, 1, 1]
const C_A = 1.75573
const C_B = 4.37917
const THETA0 = Float64[-2.74862, 0.13683, -0.583858, -3.27649, 2.03001]

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
