module Pad8fbb4fbe

const N = 5
const C_S = Float64[1.55826, 0.815201, -0.804198, 2.77652, 0.442974]
const C_V = Float64[-0.945467, -0.27003, -1.21051, -0.402696, -0.076865, -0.412015, -1.51233, -0.346899, 0.181582, -0.359892, -0.949421, -1.32119, 1.46096, -1.6186, -0.451909]
const C_D = Float64[31.6228, 5.62341, 1000, 177.828, 1]
const C_A = 3.86164
const C_B = 3.61746
const THETA0 = Float64[-4.25962, -1.20412, -0.951754, -2.39684, -0.0514831]

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
