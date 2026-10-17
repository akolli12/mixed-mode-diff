module P39f19b493f

const N = 10
const C_S = Float64[-1.53554, -0.367686, -0.967686, -3.12414, 1.26358, 1.11849, -1.39991, -0.956177, -4.04864, -1.6222]
const C_V = Float64[0.690739, 0.202399, -0.029127, -0.332632, 0.889443, -0.432331, -0.0553701, 1.37995, -0.594998, -0.579957, 0.650246, 1.12276, 0.446413, -1.32371, 1.34918, 0.0879584, -0.563198, 0.383305, 0.0613418, 0.170146, 0.244197, -0.0330194, 2.30582, 0.149651, 0.4493, -1.36285, -0.604308, -0.878777, 1.68944, 0.398955]
const C_D = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const C_A = 3.21092
const C_B = -4.00988
const THETA0 = Float64[-1.01266, -1.65766, 1.20917, -0.620204, 2.31271, -2.28785, 0.0515131, 1.11237, -2.25946, -5.20529]

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
