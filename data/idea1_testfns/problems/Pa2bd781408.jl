module Pa2bd781408

const N = 5
const C_S = Float64[2.56998, -1.08786, -1.56885, 1.38863, -0.828255]
const C_V = Float64[-1.00374, 1.55558, 0.439229, 1.68514, 0.223076, -0.0274785, 0.788114, -1.49146, -1.21586, 0.0224731, 2.0764, 1.11975, -0.0725008, -0.415586, -0.140591]
const C_D = Float64[1.77828, 3.16228, 1, 5.62341, 10]
const C_W = 0.760545
const C_A = 2.31731
const C_B = 2.35578
const THETA0 = Float64[1.75879, -0.880492, -0.949516, 1.55837, 5.39053]

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
