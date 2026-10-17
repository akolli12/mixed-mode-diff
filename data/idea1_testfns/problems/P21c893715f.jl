module P21c893715f

const N = 5
const C_S = Float64[1.48757, -1.41737, -3.17052, 2.99298, -0.288355]
const C_V = Float64[1.51147, -0.552647, 0.446253, 0.284668, 0.97206, 0.774272, 2.02089, 0.0266277, 0.462997, 1.05233, -1.077, -0.646919, 0.446327, -0.418293, 1.05885]
const C_D = Float64[10, 5.62341, 3.16228, 1.77828, 1]
const C_W = 0.596714
const C_A = 1.19657
const C_B = 4.17899
const THETA0 = Float64[-2.86379, -4.18483, -2.11504, -1.35462, -1.59371]

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
        u = abs(z[i] / C_W)
        s += C_D[i] * C_W^2 * (u + log1p(exp(-2u)) - log(2.0))
    end
    return C_A * s + C_B
end

end # module
