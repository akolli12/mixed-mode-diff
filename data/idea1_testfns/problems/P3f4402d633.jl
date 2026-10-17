module P3f4402d633

const N = 10
const C_S = Float64[-0.00319852, 1.79991, 3.11606, -2.02684, -0.610717, -0.562247, 2.04678, -1.798, 1.11204, 0.998292]
const C_V = Float64[-0.603448, 0.236884, 0.885643, 0.132282, -0.483091, -0.377676, 0.456763, 0.266225, -0.51846, 1.70441, -0.909573, -0.451281, 0.471155, 0.915811, -0.49234, -1.40184, 0.938194, 1.04179, 0.293221, -1.21468, 0.629945, 0.123587, -1.08633, -0.65845, 0.0634214, -0.613843, 0.742813, -0.558652, -0.0560758, 1.13935]
const C_D = Float64[1, 10, 2.15443, 1.29155, 4.64159, 2.78256, 3.59381, 1.6681, 5.99484, 7.74264]
const C_W = 0.344747
const C_A = 0.622655
const C_B = 2.10188
const THETA0 = Float64[-5.4417, -0.0949515, 2.19685, -1.13138, 4.50681, -1.57711, -0.895319, -5.64639, 0.288193, -5.22562]

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
