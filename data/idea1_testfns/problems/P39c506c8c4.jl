module P39c506c8c4

const N = 10
const C_S = Float64[4.39952, 0.287001, 2.27634, 2.26307, -1.72821, -2.11951, 2.31693, -2.987, 0.785021, -0.285028]
const C_V = Float64[-0.244099, 1.78156, -0.194562, -0.490976, -0.827967, 0.484088, 0.12663, -0.147121, -1.22886, 0.495553, -1.10903, 0.415688, 0.0110606, 0.427837, 0.0875373, -0.485111, 0.37258, 2.57945, 0.139484, 0.607934, -0.997476, 0.194476, -0.0216469, 0.120291, -1.12441, 0.320852, 0.957564, -0.497301, -0.137661, -0.0886058]
const C_D = Float64[7.74264, 2.78256, 21.5443, 1, 3593.81, 59.9484, 1291.55, 464.159, 10000, 166.81]
const C_A = 3.24915
const C_B = -1.65971
const THETA0 = Float64[2.7758, 0.509772, 1.01594, 3.07034, -2.02497, -3.61336, 4.17245, 1.34044, -0.324837, 1.51448]

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
