module Pcca18bb3d9

const N = 5
const C_S = Float64[-1.46591, -0.734086, 0.677051, 2.89917, 0.0649617]
const C_V = Float64[-1.6978, 1.26065, -0.0454737, -0.553317, 1.3627, -1.60469, -0.122413, 0.544951, 0.378817, 0.649706, 0.081834, 1.46401, 0.513994, 1.48399, 0.547711]
const C_D = Float64[10000, 10, 1, 1000, 100]
const C_A = 0.996504
const C_B = 0.99307
const THETA0 = Float64[-2.67062, -0.0963145, 0.328494, -1.28114, 3.02218]

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
