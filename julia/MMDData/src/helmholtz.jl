# Helmholtz with PML: mirrors python/helmholtz.py exactly (see its docstring).

function _pml_setup(c, freq, N, npml)
    nt = size(c, 1)
    h = 1.0 / N
    ω = 2π * freq
    x = [(i - npml - 0.5) * h for i in 1:nt]
    Lp = npml * h
    σ0 = 3 * log(1e6) * maximum(c) / (2Lp)
    σ(xx) = σ0 * (max(0.0, max(-xx, xx - 1.0)) / Lp)^2
    s(xx) = 1 + im * σ(xx) / ω
    return nt, h, ω, x, s
end

"""
    helmholtz_matrix(c, freq, N, npml; beta=0.0) -> SparseMatrixCSC{ComplexF64}

`c` is the sound speed on the full grid (incl. PML), size (N+2npml)^2.
`beta = 0` gives the Helmholtz operator, `beta = 0.5` the shifted-Laplacian preconditioner
(1 + 0.5im) k^2.
"""
function helmholtz_matrix(c::AbstractMatrix{<:Real}, freq, N::Integer, npml::Integer; beta = 0.0)
    nt, h, ω, x, s = _pml_setup(c, freq, N, npml)
    sn = s.(x)
    sface(i) = i == 0 ? s(x[1] - 0.5h) : s(x[i] + 0.5h)   # face between node i and i+1
    idx(i, j) = i + nt * (j - 1)
    I = Int[]; J = Int[]; V = ComplexF64[]
    @inbounds for j in 1:nt, i in 1:nt
        aw = sn[j] / sface(i - 1) / h^2
        ae = sn[j] / sface(i) / h^2
        aso = sn[i] / sface(j - 1) / h^2
        an = sn[i] / sface(j) / h^2
        p = idx(i, j)
        push!(I, p); push!(J, p)
        push!(V, aw + ae + aso + an - (1 + im * beta) * (ω / c[i, j])^2 * sn[i] * sn[j])
        if i > 1;  push!(I, p); push!(J, idx(i - 1, j)); push!(V, -aw); end
        if i < nt; push!(I, p); push!(J, idx(i + 1, j)); push!(V, -ae); end
        if j > 1;  push!(I, p); push!(J, idx(i, j - 1)); push!(V, -aso); end
        if j < nt; push!(I, p); push!(J, idx(i, j + 1)); push!(V, -an); end
    end
    return sparse(I, J, V, nt^2, nt^2)
end

function helmholtz_rhs(f::AbstractMatrix, c, freq, N, npml)
    nt, h, ω, x, s = _pml_setup(c, freq, N, npml)
    sn = s.(x)
    return vec(sn .* transpose(sn) .* f)
end

"""
    load_helmholtz(path) -> Dict{Int,NamedTuple}

freq => (; freq, N, npml, c (n,nt,nt), f (n,nt,nt)) for one `helmholtz_<medium>_<split>.h5`.
"""
function load_helmholtz(path::AbstractString)
    out = Dict{Int,Any}()
    h5open(path, "r") do h
        for name in keys(h)
            startswith(name, "freq_") || continue
            g = h[name]
            F = Int(pyattr(g, "freq"))
            out[F] = (; freq = F, N = Int(pyattr(g, "N")), npml = Int(pyattr(g, "npml")),
                      c = pyread(g["c"]), f = pyread(g["f"]))
        end
    end
    return out
end

"helmholtz_instance(hs, freq, s; beta=0) -> (A, b, c)"
function helmholtz_instance(hs, freq::Integer, s::Integer; beta = 0.0)
    g = hs[freq]
    c = g.c[s, :, :]
    A = helmholtz_matrix(c, freq, g.N, g.npml; beta)
    return A, helmholtz_rhs(g.f[s, :, :], c, freq, g.N, g.npml), c
end
