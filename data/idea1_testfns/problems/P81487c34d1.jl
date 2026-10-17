module P81487c34d1

const N = 10
const C_S = Float64[1.802, -1.15936, -2.29548, -0.238568, -3.30988, 2.39261, 0.828785, -0.489723, -0.0922506, 1.17196]
const C_V = Float64[1.34187, 0.834098, -0.411465, 0.478947, 0.693948, 0.194547, 0.499741, 1.57178, -1.61956, 0.444382, 1.15632, 1.07049, 0.0165251, -0.965751, -0.535662, 0.121955, 0.952259, 0.187691, -0.396838, 0.368838, 0.292878, -1.53638, -0.979326, -0.833994, -1.39849, -0.13843, 0.715161, 0.165912, 1.08139, 1.59032]
const C_D = Float64[1.6681, 4.64159, 1, 7.74264, 12.9155, 35.9381, 59.9484, 100, 21.5443, 2.78256]
const C_W = 0.794688
const C_A = 2.77986
const C_B = 3.93244
const THETA0 = Float64[-0.914194, -2.44867, -3.90978, 6.39709, 0.0740734, 7.49829, -4.75668, -2.04878, -2.36941, -0.242689]

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
