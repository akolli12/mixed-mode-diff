module P9fda7fa322

const N = 5
const C_S = Float64[0.315975, -2.14267, 1.87116, 1.14776, 2.54673]
const C_V = Float64[0.718635, 0.223083, 0.85297, -1.17573, -0.146074, -0.74387, -1.77489, 0.604969, 0.276794, 1.10909, -0.74573, -0.0801103, 0.48642, 0.918646, 0.826009]
const C_D = Float64[3.16228, 100, 31.6228, 1, 10]
const C_A = 0.716665
const C_B = -4.24049
const THETA0 = Float64[-0.329705, -0.79309, -3.42212, 2.27593, -0.882634]

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
