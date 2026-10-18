"""
    MMDData

Loaders, assemblers and step metrics for the mixed-mode-diff project datasets.

  Idea 1  : `load_problem`, `load_hidden`, `problem_loss`, `step_metrics`
  Idea 2  : `load_diffusion`, `diffusion_instance`, `load_helmholtz`, `helmholtz_instance`
  Checks  : `selfcheck(datadir)` verifies every assembler/problem against the Python reference values

Array layout: HDF5 files were written from Python (row-major). `pyread` returns arrays in
the *Python* index order, so `A[s, i, j]` in Julia is `A[s, i, j]` in Python, and grid
(i, j) maps to unknown `i + N*(j-1)` (column-major `vec`), matching the Python assemblers.
"""
module MMDData

using HDF5, JSON3, LinearAlgebra, SparseArrays

export pyread, pyattr,
       diffusion_matrix, diffusion_rhs, binary_k, lognormal_k, load_diffusion, diffusion_instance,
       helmholtz_matrix, helmholtz_rhs, load_helmholtz, helmholtz_instance,
       load_problem, load_hidden, load_check, problem_loss, list_problems,
       cosine_to_descent, best_step, step_metrics,
       pcg!, selfcheck

"Read an HDF5 dataset written by h5py and return it in Python index order."
pyread(ds) = (A = read(ds); ndims(A) <= 1 ? A : permutedims(A, Tuple(ndims(A):-1:1)))
pyattr(obj, name) = read_attribute(obj, name)

include("diffusion.jl")
include("helmholtz.jl")
include("idea1.jl")
include("metrics.jl")
include("solvers.jl")
include("selfcheck.jl")

end # module
