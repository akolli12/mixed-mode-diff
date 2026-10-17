module P10a68ce5e6

const N = 2
const C_S = Float64[-1.04815, 1.68194]
const C_V = Float64[-0.143448, 0.119735, 0.173631, 0.414665, 2.52295, -1.51082]
const C_D = Float64[1, 1]
const C_W = 0.403642
const C_A = 3.48869
const C_B = 4.74
const THETA0 = Float64[-2.81922, -2.83114]

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
