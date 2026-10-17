module P09dc4a70b0

const N = 10
const C_S = Float64[0.415404, -0.814703, 0.756896, 0.632532, 1.45657, 2.21962, -0.977589, -1.84058, 2.44417, 0.113828]
const C_V = Float64[-0.61383, 1.0266, 0.716341, 1.7336, -0.656412, 1.59353, -0.795791, 1.14743, -0.65262, 0.407744, 0.198382, -0.244633, 0.894146, -0.893767, -0.273047, 1.13629, -0.058233, 0.353771, 0.583712, -0.132111, -0.727034, -0.90189, -1.18138, 0.566541, 0.640865, -0.0257245, -2.65669, 0.439395, -2.16585, 0.207542]
const C_D = Float64[464.159, 10000, 3593.81, 7.74264, 2.78256, 166.81, 1, 59.9484, 1291.55, 21.5443]
const C_A = 1.76094
const C_B = 3.91719
const THETA0 = Float64[0.314649, -1.39102, -1.67463, 1.01035, 2.72827, 1.84536, -0.192765, -3.31796, 1.66806, -1.10338]

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
