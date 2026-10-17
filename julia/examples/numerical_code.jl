# Plain numerical code with no knowledge of AD. Enzyme differentiates it as written.
# Every function returns a scalar loss and takes its parameters as positional arguments
# (Enzyme handles positional arguments more cleanly than keyword arguments).

# Quadratic bowl f(θ) = ½ θᵀAθ − bᵀθ. Gradient is Aθ − b, so we can check Enzyme exactly.
function quadratic_bowl(θ::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    n = length(θ)
    f = 0.0
    for i in 1:n
        Aθ_i = 0.0
        for j in 1:n
            Aθ_i += A[i, j] * θ[j]
        end
        f += 0.5 * θ[i] * Aθ_i - b[i] * θ[i]
    end
    return f
end

# n-dimensional Rosenbrock valley; minimum is 0 at θ = (1, …, 1).
function rosenbrock(θ::Vector{Float64})
    f = 0.0
    for i in 1:length(θ)-1
        f += 100.0 * (θ[i+1] - θ[i]^2)^2 + (1.0 - θ[i])^2
    end
    return f
end

# Successive over-relaxation (SOR) for Ax = b, run for a fixed number of sweeps from x = 0.
# Returns log10 of the relative error against a known solution xstar.
# The solution vector is updated in place, which Enzyme supports and Zygote does not.
function sor_loss(ω::Float64, A::Matrix{Float64}, b::Vector{Float64},
                  xstar::Vector{Float64}, iters::Int)
    n = length(b)
    x = zeros(n)
    for _ in 1:iters
        for i in 1:n
            σ = 0.0
            for j in 1:n
                if j != i
                    σ += A[i, j] * x[j]
                end
            end
            x[i] = (1.0 - ω) * x[i] + ω * (b[i] - σ) / A[i, i]
        end
    end
    err = 0.0
    ref = 0.0
    for i in 1:n
        err += (x[i] - xstar[i])^2
        ref += xstar[i]^2
    end
    return 0.5 * log10(err / ref)
end

# 1-D Poisson matrix (tridiagonal 2, −1), the standard SOR test problem.
function poisson_1d(n::Int)
    A = zeros(n, n)
    for i in 1:n
        A[i, i] = 2.0
        i > 1 && (A[i, i-1] = -1.0)
        i < n && (A[i, i+1] = -1.0)
    end
    return A
end
