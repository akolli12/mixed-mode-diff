module Pcb634c42ab

const N = 10
const C_S = Float64[0.844099, 1.39861, -2.33096, -3.35336, -0.0591464, 0.946715, 2.13761, -1.67962, 2.51044, -0.909868]
const C_V = Float64[1.31062, 2.12907, 0.666169, 0.163291, 0.861635, 0.78143, -0.493509, -1.1613, 1.77874, -1.55803, 0.484702, -0.572888, 0.00275977, 1.08921, -1.36305, -0.397115, 0.127973, 0.504881, -1.02468, -1.1476, 0.98005, -1.39214, -0.093006, 0.955835, 0.438433, -0.678038, -0.00651161, -0.159406, 0.692727, 1.59237]
const C_D = Float64[2.78256, 4.64159, 2.15443, 10, 3.59381, 1.29155, 1, 7.74264, 1.6681, 5.99484]
const C_W = 0.506628
const C_A = 4.2621
const C_B = 3.76071
const THETA0 = Float64[4.30924, -0.964008, -1.17745, 3.14103, -2.52969, -2.0085, -1.33624, 4.16798, -4.67903, 1.10026]

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
