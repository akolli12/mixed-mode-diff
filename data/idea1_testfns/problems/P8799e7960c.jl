module P8799e7960c

const N = 10
const C_S = Float64[-1.0046, -1.4078, -1.138, 0.0988939, -3.29754, 0.44368, -1.14882, 1.42981, 3.8535, 0.770727]
const C_V = Float64[-1.12317, -0.0259819, 0.274054, 0.566741, 0.467451, -1.3913, 0.231323, 1.29039, -0.214414, 0.329619, 1.40748, -0.189303, 3.00913, -0.197879, -0.601179, -0.129147, -0.180187, -1.85114, -1.00889, -1.0263, 1.44829, -0.301906, 1.50556, -0.415645, -0.415547, -0.868881, 1.66087, 1.64902, 0.908597, 0.839691]
const C_D = Float64[215.443, 2.15443, 1, 1000, 464.159, 46.4159, 21.5443, 10, 4.64159, 100]
const C_A = 1.35539
const C_B = -3.80214
const THETA0 = Float64[-2.36076, -1.06818, 0.497433, 0.961683, -4.68228, -1.77898, -0.867692, 1.76643, 1.9712, 0.840361]

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
