module P19530206c6

const N = 5
const C_S = Float64[1.34733, 1.2229, 3.70094, -0.271537, -2.78525]
const C_V = Float64[0.752512, -0.963639, 0.271587, -0.433872, 2.06202, 0.808344, 1.77672, -0.402722, 0.827764, 0.228596, 0.631697, -0.996025, -0.930311, 2.57515, 1.61995]
const C_D = Float64[1000, 177.828, 1, 31.6228, 5.62341]
const C_W = 0.978266
const C_A = 2.55051
const C_B = 0.277203
const THETA0 = Float64[3.53484, -5.16147, 1.3804, 6.88112, -5.26661]

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
