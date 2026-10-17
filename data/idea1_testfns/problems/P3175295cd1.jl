module P3175295cd1

const N = 10
const C_S = Float64[1.56028, 0.714002, -2.25274, 2.45329, -4.52282, -1.31672, -1.60599, -0.0248416, 1.65686, -2.88843]
const C_V = Float64[0.624051, 0.210348, -1.12978, -0.361467, -0.263172, 0.295103, -0.461212, 0.35491, 0.268028, 1.24595, 0.584804, -1.26939, 0.669601, 0.168431, 1.51873, -0.561366, -0.861459, 0.686672, 0.100565, 0.68369, 0.175714, 1.63626, 0.807162, -1.65017, -0.865255, 0.9883, 2.40276, -0.613303, 1.1079, 0.833001]
const C_D = Float64[100, 464.159, 2.15443, 10, 215.443, 1000, 46.4159, 21.5443, 1, 4.64159]
const C_W = 0.525121
const C_A = 0.641325
const C_B = -0.819706
const THETA0 = Float64[5.7889, -4.10966, 6.42, 7.79496, 2.69656, -4.7872, -4.10738, -1.17131, 0.30739, 1.21391]

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
        u = abs(z[i] / C_W)
        s += C_D[i] * C_W^2 * (u + log1p(exp(-2u)) - log(2.0))
    end
    return C_A * s + C_B
end

end # module
