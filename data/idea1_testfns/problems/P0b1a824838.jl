module P0b1a824838

const N = 10
const C_S = Float64[2.31367, 1.62788, 0.0591556, 0.6679, 0.000103952, 0.635438, 0.377846, -2.25254, -0.0775875, -2.18969]
const C_V = Float64[-0.42093, 0.41921, -1.12581, -0.209883, -0.964537, 1.16059, 1.25866, -0.673365, 1.57822, 0.37438, 0.74749, 1.09031, -0.450414, 0.283133, -1.45979, -1.83383, -0.775885, -0.980904, 0.309343, -0.861371, 1.05504, -0.719718, -1.13947, 0.612631, 1.13961, 0.607798, -0.0558831, -1.1005, 0.338572, 1.13152]
const C_D = Float64[12.9155, 35.9381, 4.64159, 7.74264, 1, 1.6681, 2.78256, 100, 21.5443, 59.9484]
const C_A = 4.71905
const C_B = 2.03808
const THETA0 = Float64[0.666478, 1.03084, 1.34067, -0.179593, 1.0212, 2.73818, 2.74398, -1.26018, 0.336294, 0.330978]

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
