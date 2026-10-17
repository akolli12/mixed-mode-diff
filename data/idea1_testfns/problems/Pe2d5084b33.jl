module Pe2d5084b33

const N = 5
const C_S = Float64[-0.358002, 2.85221, 0.509454, 0.0296109, 1.25233]
const C_V = Float64[-0.681385, -0.351132, 1.19743, 0.724748, 0.808689, 0.445189, -0.616726, -0.819982, 2.06248, 0.330694, -0.960589, 0.362809, 0.133563, 0.861787, -1.44193]
const C_D = Float64[1, 1, 1, 1, 1]
const C_W = 0.579323
const C_A = 0.582675
const C_B = 0.320214
const THETA0 = Float64[0.130054, 1.93471, -1.51744, -5.27339, -1.95912]

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
