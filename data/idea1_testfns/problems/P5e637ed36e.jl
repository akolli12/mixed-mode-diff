module P5e637ed36e

const N = 5
const C_S = Float64[-1.36377, 0.739883, -2.37596, 0.371749, 1.71735]
const C_V = Float64[-0.481257, 0.356461, 0.491328, -0.107964, 0.396673, 0.352268, 1.17999, 0.776583, -0.318126, -0.19342, 0.840086, 0.506639, -0.480501, 0.827083, 0.365094]
const C_D = Float64[1, 1000, 31.6228, 5.62341, 177.828]
const C_A = 1.02022
const C_B = -3.22908
const THETA0 = Float64[1.88668, 0.287476, -2.96068, 1.00398, -2.06701]

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
