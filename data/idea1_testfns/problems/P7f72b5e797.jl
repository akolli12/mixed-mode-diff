module P7f72b5e797

const N = 2
const C_S = Float64[0.301828, 1.89114]
const C_V = Float64[0.052021, 0.877854, -0.388796, 0.067408, -1.19491, 2.03274]
const C_D = Float64[1, 1000]
const C_A = 0.537234
const C_B = 4.8077
const THETA0 = Float64[-1.65745, 1.63146]

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
