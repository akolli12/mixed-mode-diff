module P79053eb988

const N = 2
const C_S = Float64[-0.282836, -2.30938]
const C_V = Float64[-0.433016, 0.96695, -0.259208, 1.07524, 0.0544206, 0.105875]
const C_D = Float64[10000, 1]
const C_A = 1.32173
const C_B = 2.88345
const THETA0 = Float64[-0.368956, 1.88101]

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
