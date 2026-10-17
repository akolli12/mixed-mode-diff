module P1e15e1fae2

const N = 10
const C_S = Float64[0.269028, -0.258725, 3.59923, 0.194421, 1.62997, -0.649802, 1.7905, -1.12373, -4.79241, 0.637577]
const C_V = Float64[0.157408, -0.843826, 0.507734, 0.703783, 0.478107, -0.411482, -1.07469, -1.13381, 0.556807, -2.24869, 0.510016, 1.87322, -0.965657, -1.18588, -1.72797, 1.07974, -0.424876, 0.214634, 0.283329, -0.840304, 1.34212, 0.292428, 0.607611, -0.11265, 1.05275, 1.36957, -0.0326408, 0.0782398, 1.36457, -0.905492]
const C_D = Float64[12.9155, 35.9381, 2.78256, 100, 59.9484, 21.5443, 7.74264, 1, 4.64159, 1.6681]
const C_W = 0.785023
const C_A = 1.71572
const C_B = -2.17042
const THETA0 = Float64[0.298509, 8.27473, 0.906654, -7.15367, -3.36925, 1.84, -0.347725, -2.12228, -4.74996, -3.21913]

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
