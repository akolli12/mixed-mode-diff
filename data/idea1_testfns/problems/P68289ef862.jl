module P68289ef862

const N = 5
const C_S = Float64[-0.806184, -3.96461, 0.676446, -0.00751057, 0.147078]
const C_V = Float64[-0.417824, 0.392719, -0.0154297, -0.0534615, -1.99957, 2.74167, 0.445717, 0.660912, -0.822657, -0.772482, 0.20341, 0.550544, -2.01537, -0.816323, 0.871767]
const C_D = Float64[100, 1, 1000, 10, 10000]
const C_W = 0.444426
const C_A = 1.79946
const C_B = 2.89464
const THETA0 = Float64[-0.477443, -5.45738, 2.44727, 0.482338, 1.51292]

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
