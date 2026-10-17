module P91f8b08627

const N = 10
const C_S = Float64[-1.82753, 1.03653, -1.0954, 1.89632, 0.318168, -0.131073, 1.60242, 2.49953, -1.52944, 0.600962]
const C_V = Float64[-1.09096, -0.0617442, 0.261195, 2.06207, 1.44165, 0.0423044, -1.12516, -1.46819, -0.657731, -0.643629, -0.215871, 0.355977, -0.498266, -1.48881, -0.842227, 0.455986, 0.146591, -0.845329, 0.94013, 0.952106, 1.56362, 0.953422, -1.73942, -1.30372, -0.618241, 1.60224, -0.823086, 0.0557023, 0.827101, 1.10594]
const C_D = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const C_A = 1.19399
const C_B = 3.24928
const THETA0 = Float64[-3.43456, 1.08096, -2.78323, 0.25573, 0.0512535, 0.235749, 1.20676, 0.682879, -1.06937, 1.40592]

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
