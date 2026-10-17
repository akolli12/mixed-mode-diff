module P88ce4779f8

const N = 10
const C_S = Float64[-0.819742, -0.14988, 0.306476, -1.70615, -3.55891, -1.57508, -0.485332, 3.17783, 0.270896, -1.05349]
const C_V = Float64[0.73829, -1.52331, -1.39736, 1.65997, 1.06229, -0.754481, 1.09313, 0.523049, 0.207899, 0.43563, 1.31993, -0.868715, -0.813162, 0.176197, 0.0773201, 0.192035, -1.17685, 1.13872, -1.05423, 0.198949, -0.232594, 1.12457, -1.22002, 1.07025, 1.14919, -0.0464112, -1.08363, -0.0168469, -1.28827, 1.77303]
const C_SC = Float64[1, 59.9484, 7.74264, 2.78256, 35.9381, 100, 21.5443, 12.9155, 4.64159, 1.6681]
const C_FR = Float64[3.29044, 4.41122, 20.5665, 6.0637, 3.60544, 29.0702, 16.5078, 25.6898, 11.4796, 15.9446]
const C_PH = Float64[2.2134, 4.08963, 0.502495, 2.45477, 1.08311, 5.9266, 1.86342, 3.60607, 0.904551, 6.05141]
const C_Y = Float64[-1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1]
const C_A = 0.666856
const C_B = -2.59468
const THETA0 = Float64[-2.65253, 1.17522, -2.06193, 1.89064, -0.604106, -2.91147, 1.28038, 1.94852, 0.266036, 1.33885]

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
