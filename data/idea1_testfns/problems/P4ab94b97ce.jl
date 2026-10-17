module P4ab94b97ce

const N = 5
const C_S = Float64[2.93667, 0.494464, -1.84915, 1.22055, 1.96524]
const C_V = Float64[-0.535928, 0.399668, 1.85449, 0.755836, -0.000539733, 1.81454, -0.734852, -0.690028, -0.934364, -1.37644, 0.219879, -0.827365, -0.286906, 0.324044, -0.679405]
const C_D = Float64[1, 1, 1, 1, 1]
const C_W = 0.86024
const C_A = 3.42445
const C_B = 2.40896
const THETA0 = Float64[-0.488153, 1.3753, -6.13485, 1.3916, 5.17683]

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
