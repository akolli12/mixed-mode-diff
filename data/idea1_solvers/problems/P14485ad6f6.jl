module P14485ad6f6

const N = 16
include(joinpath(@__DIR__, "..", "constants", "P14485ad6f6.jl"))

function loss(θ::AbstractVector{T}) where {T<:Real}
    G = C_G
    k = Matrix{Float64}(undef, G, G)
    @inbounds for j in 1:G, i in 1:G
        k[i, j] = exp(C_AMP * sin(2π * C_F1 * i / (G + 1) + C_P1) * sin(2π * C_F2 * j / (G + 1) + C_P2))
    end
    cw = similar(k); ce = similar(k); cs = similar(k); cn = similar(k)
    @inbounds for j in 1:G, i in 1:G
        cw[i, j] = i > 1 ? 2k[i-1, j] * k[i, j] / (k[i-1, j] + k[i, j]) : k[i, j]
        ce[i, j] = i < G ? 2k[i+1, j] * k[i, j] / (k[i+1, j] + k[i, j]) : k[i, j]
        cs[i, j] = j > 1 ? 2k[i, j-1] * k[i, j] / (k[i, j-1] + k[i, j]) : k[i, j]
        cn[i, j] = j < G ? 2k[i, j+1] * k[i, j] / (k[i, j+1] + k[i, j]) : k[i, j]
    end
    b = Matrix{Float64}(undef, G, G)
    @inbounds for j in 1:G, i in 1:G
        b[i, j] = 1 + 0.5 * sin(C_R1 * i / (G + 1)) * cos(C_R2 * j / (G + 1))
    end
    bnorm = sqrt(sum(abs2, b))
    x = zeros(T, G, G)
    Ax = zeros(T, G, G)
    for it in 1:N
        @inbounds for j in 1:G, i in 1:G
            v = (cw[i, j] + ce[i, j] + cs[i, j] + cn[i, j]) * x[i, j]
            if i > 1; v -= cw[i, j] * x[i-1, j]; end
            if i < G; v -= ce[i, j] * x[i+1, j]; end
            if j > 1; v -= cs[i, j] * x[i, j-1]; end
            if j < G; v -= cn[i, j] * x[i, j+1]; end
            Ax[i, j] = v
        end
        @inbounds for j in 1:G, i in 1:G
            x[i, j] += θ[it] * (b[i, j] - Ax[i, j])
        end
    end
    r = zeros(T, G, G)
    @inbounds for j in 1:G, i in 1:G
        v = (cw[i, j] + ce[i, j] + cs[i, j] + cn[i, j]) * x[i, j]
        if i > 1; v -= cw[i, j] * x[i-1, j]; end
        if i < G; v -= ce[i, j] * x[i+1, j]; end
        if j > 1; v -= cs[i, j] * x[i, j-1]; end
        if j < G; v -= cn[i, j] * x[i, j+1]; end
        r[i, j] = b[i, j] - v
    end
    s = zero(T)
    @inbounds for j in 1:G, i in 1:G
        s += r[i, j]^2
    end
    return log10(sqrt(s) / bnorm)
end

end # module
