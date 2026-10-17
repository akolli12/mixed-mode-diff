module Pa89c52d06c

const N = 10
const C_S = Float64[-0.330744, 1.69745, 0.0928232, -2.78565, 1.98579, 3.0979, 1.62737, 0.965956, -2.89983, -0.855597]
const C_V = Float64[-0.0920711, 0.259894, 1.01401, 0.387442, -0.896481, 0.183863, -1.16333, 0.952048, 0.428129, 0.903821, 1.34117, -0.773495, -1.27575, 0.815836, 0.429981, -0.177846, -0.417923, -0.405436, 0.096893, -0.137115, 0.323541, -1.35296, -0.315941, -0.265156, -1.40825, -0.514344, -0.327907, -1.19837, -0.842961, -0.727019]
const C_D = Float64[2.15443, 1000, 21.5443, 4.64159, 10, 215.443, 46.4159, 1, 464.159, 100]
const C_A = 3.15267
const C_B = 1.16482
const THETA0 = Float64[0.824034, -0.368354, 2.85928, -2.48574, -4.0412, 1.9199, -2.99715, 2.20294, -1.9546, -0.348258]

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
