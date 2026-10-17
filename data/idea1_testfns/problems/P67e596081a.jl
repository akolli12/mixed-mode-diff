module P67e596081a

const N = 5
const C_S = Float64[-1.00922, 1.33336, 2.33124, 0.962722, -1.44729]
const C_V = Float64[0.617862, 1.27903, -0.316342, 1.84526, -1.46705, -1.21898, 1.14035, -1.24775, -1.52949, 0.890407, -0.606535, 0.542214, -0.398812, 0.127147, 0.700656]
const C_D = Float64[1, 1, 1, 1, 1]
const C_W = 0.301331
const C_A = 4.24169
const C_B = -3.98013
const THETA0 = Float64[-0.896202, 6.65279, 2.73695, -4.03115, 1.78125]

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
