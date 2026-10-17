# Differentiates the functions in numerical_code.jl with Enzyme and checks every
# gradient against central finite differences.
#
# Run from the repo root:
#   julia --project=julia julia/examples/enzyme_demo.jl

using Enzyme
using LinearAlgebra
using Random
using Printf

include("numerical_code.jl")

# Central finite-difference gradient, used only as an independent check on Enzyme.
function fd_gradient(f, θ::Vector{Float64}; h=1e-6)
    g = similar(θ)
    for i in eachindex(θ)
        θp = copy(θ); θp[i] += h
        θm = copy(θ); θm[i] -= h
        g[i] = (f(θp) - f(θm)) / (2h)
    end
    return g
end

relerr(a, b) = norm(a - b) / max(norm(b), eps())

function report(name, g_enzyme, g_check, check_name)
    @printf("%-16s rel. error vs %-15s %.2e\n", name, check_name, relerr(g_enzyme, g_check))
end

Random.seed!(0)
n = 10

# 1. Quadratic bowl: vector parameters, reverse mode.
# Duplicated(θ, dθ) gives Enzyme the input and a zeroed buffer to accumulate ∂f/∂θ into.
# Const marks arguments we do not differentiate with respect to.
M = randn(n, n)
A = M' * M + I                        # symmetric positive definite
b = randn(n)
θ = randn(n)

dθ = zeros(n)
autodiff(Reverse, quadratic_bowl, Active, Duplicated(θ, dθ), Const(A), Const(b))
report("quadratic_bowl", dθ, A * θ - b, "exact (Aθ − b)")
report("quadratic_bowl", dθ, fd_gradient(t -> quadratic_bowl(t, A, b), θ), "finite diff")

# 2. Rosenbrock: same pattern, nonconvex.
θ = randn(n)
dθ = zeros(n)
autodiff(Reverse, rosenbrock, Active, Duplicated(θ, dθ))
report("rosenbrock", dθ, fd_gradient(rosenbrock, θ), "finite diff")

# 3. SOR: derivative of the solver's error with respect to its relaxation factor ω.
# A scalar input is marked Active; reverse mode returns its derivative directly.
m = 50
A = poisson_1d(m)
xstar = randn(m)
b = A * xstar
iters = 20

println("\nSOR on 1-D Poisson (n = $m, $iters sweeps): loss = log10 relative error")
println("   ω      loss    dloss/dω (Enzyme)   dloss/dω (finite diff)")
for ω in (1.0, 1.3, 1.6, 1.8, 1.9)
    ((dω, _, _, _, _),) = autodiff(Reverse, sor_loss, Active,
                                   Active(ω), Const(A), Const(b), Const(xstar), Const(iters))
    h = 1e-6
    fd = (sor_loss(ω + h, A, b, xstar, iters) - sor_loss(ω - h, A, b, xstar, iters)) / (2h)
    @printf("  %.2f  %7.3f   %+.6e      %+.6e\n", ω, sor_loss(ω, A, b, xstar, iters), dω, fd)
end

# 4. Gradient descent on ω using Enzyme's derivative: a first taste of AD-driven tuning.
ω = 1.0
lr = 0.05
for step in 1:30
    global ω
    ((dω, _, _, _, _),) = autodiff(Reverse, sor_loss, Active,
                                   Active(ω), Const(A), Const(b), Const(xstar), Const(iters))
    ω = clamp(ω - lr * dω, 0.05, 1.95)   # SOR converges only for 0 < ω < 2
end
@printf("\nGradient descent on ω: ω = %.3f, loss = %.3f (ω = 1, i.e. Gauss–Seidel: %.3f)\n",
        ω, sor_loss(ω, A, b, xstar, iters), sor_loss(1.0, A, b, xstar, iters))
