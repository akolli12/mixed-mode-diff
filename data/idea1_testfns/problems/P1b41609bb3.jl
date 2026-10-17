module P1b41609bb3

const N = 5
const C_S = Float64[-0.648089, 0.606576, 2.27555, -1.32103, -0.989403]
const C_V = Float64[-0.484877, -0.430242, -0.552691, 0.654354, 1.20303, 0.113757, -1.62946, -0.891489, -0.58884, -1.1511, 1.24013, -0.0825482, -0.343596, -1.34274, 1.41177]
const C_D = Float64[5.62341, 1.77828, 10, 3.16228, 1]
const C_A = 2.02292
const C_B = 2.56646
const THETA0 = Float64[-0.63453, -2.96718, -0.844377, 0.159876, 1.59574]

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
