module P5575c6d617

const N = 2
const C_S = Float64[-0.332764, 1.8537]
const C_V = Float64[0.442888, -0.033339, 0.319176, -0.456298, 0.639133, 0.2417]
const C_D = Float64[100, 1]
const C_A = 2.7569
const C_B = -3.66427
const THETA0 = Float64[-2.65361, -0.653942]

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
