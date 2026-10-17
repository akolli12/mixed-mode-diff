module P5f30e0ae31

const N = 5
const C_S = Float64[-2.84756, -2.22794, 1.50285, 0.276711, 1.37108]
const C_V = Float64[-1.86041, -0.11523, -0.292892, -0.57602, 0.447925, 1.27399, -0.875591, -1.04824, 0.461736, -0.162607, -3.55791, 1.28164, -0.483742, 0.132852, -0.0355908]
const C_D = Float64[1.77828, 5.62341, 3.16228, 10, 1]
const C_A = 0.79414
const C_B = 2.75351
const THETA0 = Float64[-0.757989, -3.64728, 2.41267, 1.06584, 2.41715]

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
