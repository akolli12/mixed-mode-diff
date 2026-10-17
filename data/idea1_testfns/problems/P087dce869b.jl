module P087dce869b

const N = 5
const C_S = Float64[1.43573, 2.05545, -1.67859, 0.0793005, 2.11809]
const C_V = Float64[1.44838, -1.32309, 1.01627, 1.13939, 0.398645, 1.029, 0.45714, 0.60889, -0.137555, -0.625353, -0.827923, 0.523298, -1.01582, 0.596926, -1.5185]
const C_D = Float64[1, 31.6228, 1000, 5.62341, 177.828]
const C_A = 0.880624
const C_B = 4.21139
const THETA0 = Float64[2.4254, -0.285411, -2.3443, 2.77449, 0.887042]

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
