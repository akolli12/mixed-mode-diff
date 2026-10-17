module P35663c049d

const N = 10
const C_S = Float64[-2.85086, 1.8141, 0.626925, 1.08225, 2.56858, 1.0952, -0.153311, -1.77172, 1.55105, 0.750275]
const C_V = Float64[0.432255, -0.399009, -0.662255, -0.852254, 0.417141, -1.63443, 0.73188, 0.225456, 2.06443, 0.236349, 0.208732, 0.307272, 0.804719, 0.0858811, -0.114332, -1.48968, 0.846043, 0.504724, 0.482691, 0.913489, -1.22787, -1.76404, -0.600265, -0.999554, 1.42245, 0.0767269, -1.19754, 0.876283, 2.2224, 0.787666]
const C_SC = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const C_FR = Float64[9.01295, 8.01611, 9.51909, 16.4421, 17.5623, 28.8842, 2.27006, 15.3681, 2.12998, 15.6375]
const C_PH = Float64[1.89328, 4.76461, 6.12365, 3.37391, 0.205905, 4.94562, 1.51666, 2.6561, 3.8917, 6.16332]
const C_Y = Float64[1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
const C_A = 1.17509
const C_B = -1.53191
const THETA0 = Float64[-1.90586, 0.900105, 1.56298, 2.26136, 1.34045, -0.248621, 1.23438, -3.68182, 1.26761, 0.862646]

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
    w = z
    M = length(C_Y)
    s = zero(T)
    @inbounds for i in 1:M
        t = i / M
        dot = zero(T)
        for j in 1:N
            dot += C_SC[j] * cos(C_FR[j] * t + C_PH[j]) * w[j]
        end
        u = -C_Y[i] * dot
        s += u > 0 ? u + log1p(exp(-u)) : log1p(exp(u))
    end
    r = zero(T)
    @inbounds for j in 1:N
        r += w[j]^2
    end
    return C_A * (s / M + 0.5 * 0.01 * r) + C_B
end

end # module
