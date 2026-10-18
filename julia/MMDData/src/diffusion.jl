# High-contrast diffusion: mirrors python/diffusion.py exactly.
#   cell-centred FV on N x N, h = 1/N, harmonic interior faces, boundary transmissibility 2k,
#   A u = f / N^2.

"""
    diffusion_matrix(k) -> SparseMatrixCSC

5-point finite-volume matrix for coefficient `k` (N x N). Unknown (i,j) -> i + N*(j-1).
"""
function diffusion_matrix(k::AbstractMatrix{<:Real})
    N = size(k, 1)
    @assert size(k, 2) == N
    hm(a, b) = 2a * b / (a + b)
    idx(i, j) = i + N * (j - 1)
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 5N^2); sizehint!(J, 5N^2); sizehint!(V, 5N^2)
    @inbounds for j in 1:N, i in 1:N
        tw = i > 1 ? hm(k[i-1, j], k[i, j]) : 2k[i, j]
        te = i < N ? hm(k[i+1, j], k[i, j]) : 2k[i, j]
        ts = j > 1 ? hm(k[i, j-1], k[i, j]) : 2k[i, j]
        tn = j < N ? hm(k[i, j+1], k[i, j]) : 2k[i, j]
        p = idx(i, j)
        push!(I, p); push!(J, p); push!(V, tw + te + ts + tn)
        if i > 1; push!(I, p); push!(J, idx(i - 1, j)); push!(V, -tw); end
        if i < N; push!(I, p); push!(J, idx(i + 1, j)); push!(V, -te); end
        if j > 1; push!(I, p); push!(J, idx(i, j - 1)); push!(V, -ts); end
        if j < N; push!(I, p); push!(J, idx(i, j + 1)); push!(V, -tn); end
    end
    return sparse(I, J, V, N^2, N^2)
end

diffusion_rhs(f::AbstractMatrix) = vec(f) ./ size(f, 1)^2

"k = 1 + (10^(sign*p) - 1) * mask  (binary families: channels, inclusions, barriers)"
binary_k(mask, p, sign) = 1.0 .+ (10.0^(sign * p) - 1.0) .* Float64.(mask)

"k = exp(sqrt(sigma2) * phi)  (lognormal families)"
lognormal_k(phi, sigma2) = exp.(sqrt(sigma2) .* phi)

"""
    load_diffusion(path) -> NamedTuple

Reads one `diffusion_<family>_N<N>_<split>.h5` file fully into memory.
Fields: family, N, split, f (n,N,N), and either mask + contrast_exponents + sign
(binary families) or phi + sigma2 (lognormal families).
"""
function load_diffusion(path::AbstractString)
    h5open(path, "r") do h
        fam = pyattr(h, "family")
        base = (; path, family = fam, N = Int(pyattr(h, "N")), split = pyattr(h, "split"),
                f = pyread(h["f"]))
        if startswith(fam, "lognormal")
            return (; base..., phi = pyread(h["phi"]), sigma2 = collect(pyattr(h, "sigma2")))
        else
            return (; base..., mask = pyread(h["mask"]), contrast_exponents = collect(pyattr(h, "contrast_exponents")),
                    sign = Int(pyattr(h, "sign")))
        end
    end
end

"""
    diffusion_instance(ds, s; p=nothing, sigma2=nothing) -> (A, b, k)

Instance `s` (1-based) at contrast exponent `p` (binary families) or variance `sigma2`
(lognormal families).
"""
function diffusion_instance(ds, s::Integer; p = nothing, sigma2 = nothing)
    k = if hasproperty(ds, :mask)
        p === nothing && error("binary family: pass p (log10 contrast)")
        binary_k(ds.mask[s, :, :], p, ds.sign)
    else
        sigma2 === nothing && error("lognormal family: pass sigma2")
        lognormal_k(ds.phi[s, :, :], sigma2)
    end
    return diffusion_matrix(k), diffusion_rhs(ds.f[s, :, :]), k
end
