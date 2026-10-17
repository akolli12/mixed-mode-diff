module P21172a81ae

const N = 10
const C_S = Float64[3.17793, -1.01879, 0.576027, -1.28866, 1.60285, -1.45651, 2.87784, 0.525817, 0.756496, 1.40248]
const C_V = Float64[0.464993, 0.768428, -2.86745, 0.746621, -0.142379, -0.234, -0.547987, 1.70225, -0.816197, 0.48783, -0.898911, -1.09649, 0.0438812, 0.435273, 0.985513, -0.977502, -1.07893, 0.901478, -0.806846, 0.338535, 1.48073, -0.0921585, -0.59408, -1.15149, -2.9302, -0.220675, 2.03251, -1.33811, -1.775, 0.292577]
const C_D = Float64[1.29155, 1.6681, 2.78256, 4.64159, 7.74264, 2.15443, 10, 5.99484, 1, 3.59381]
const C_W = 0.861307
const C_A = 4.71159
const C_B = 2.12101
const THETA0 = Float64[4.02649, -0.792795, 0.84848, -5.74455, -0.670276, -4.59181, 4.28901, 5.40825, 1.54532, -2.65057]

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
