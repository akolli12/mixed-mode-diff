module Pfe62c54fd8

const N = 2
const C_S = Float64[-0.592143, -3.58548]
const C_V = Float64[0.896364, 0.403662, -0.962264, -0.738761, 0.585429, -1.16297]
const C_D = Float64[1, 1000]
const C_A = 3.55406
const C_B = -1.41028
const THETA0 = Float64[-1.0712, 3.84388]

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
