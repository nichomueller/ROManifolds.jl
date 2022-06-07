include("MV_snapshots.jl")
#= function M_DEIM_POD(S::SparseMatrixCSC, ϵ = 1e-5)

  S̃ = copy(S)
  row_idx, _, _ = findnz(S̃)
  unique!(row_idx)
  S̃ = Matrix(S̃[row_idx, :])

  M_DEIM_mat, Σ, _ = svd(S̃)

  total_energy = sum(Σ .^ 2)
  cumulative_energy = 0.0
  N = 0
  M_DEIM_err_bound = 1e5
  nₛ = size(S̃)[2]
  crit_val = norm(inv(M_DEIM_mat'M_DEIM_mat))
  mult_factor = sqrt(nₛ) * crit_val

  while N ≤ size(S̃)[2]-2 && (M_DEIM_err_bound > ϵ ||
    cumulative_energy / total_energy < 1.0 - ϵ ^ 2)
    N += 1
    cumulative_energy += Σ[N] ^ 2
    M_DEIM_err_bound = Σ[N + 1] * mult_factor
    @info "(M)DEIM-POD loop number $N, projection error = $M_DEIM_err_bound"
  end
  M_DEIM_mat = M_DEIM_mat[:, 1:N]
  _, col_idx, val = findnz(sparse(M_DEIM_mat))
  sparse_M_DEIM_mat = sparse(repeat(row_idx, N), col_idx, val, size(S̃)[1], N)

  @info "Basis number obtained via POD is $N, projection error ≤ $M_DEIM_err_bound"

  return sparse_M_DEIM_mat, Σ

end =#

function M_DEIM_POD(S::Matrix, ϵ = 1e-5)

  S̃ = copy(S)
  M_DEIM_mat, Σ, _ = svd(S̃)

  total_energy = sum(Σ .^ 2)
  cumulative_energy = 0.0
  N = 0
  M_DEIM_err_bound = 1e5
  nₛ = size(S̃)[2]
  crit_val = norm(inv(M_DEIM_mat'M_DEIM_mat))
  mult_factor = sqrt(nₛ) * crit_val

  while N ≤ size(S̃)[2]-2 && (M_DEIM_err_bound > ϵ ||
    cumulative_energy / total_energy < 1.0 - ϵ ^ 2)
    N += 1
    cumulative_energy += Σ[N] ^ 2
    M_DEIM_err_bound = Σ[N + 1] * mult_factor
    @info "(M)DEIM-POD loop number $N, projection error = $M_DEIM_err_bound"
  end
  M_DEIM_mat = M_DEIM_mat[:, 1:N]

  @info "Basis number obtained via POD is $N, projection error ≤ $M_DEIM_err_bound"

  return M_DEIM_mat, Σ

end

#= function M_DEIM_offline(sparse_M_DEIM_mat::SparseMatrixCSC, Σ::Vector)

  row_idx, _, _ = findnz(sparse_M_DEIM_mat)
  unique!(row_idx)
  M_DEIM_mat = Matrix(sparse_M_DEIM_mat[row_idx, :])

  (N, n) = size(M_DEIM_mat)
  n_new = n
  M_DEIM_idx = Int64[]
  append!(M_DEIM_idx, convert(Int64, argmax(abs.(M_DEIM_mat[:, 1]))[1]))
  for m in range(2, n)
    res = (M_DEIM_mat[:, m] -
    M_DEIM_mat[:, 1:(m-1)] * (M_DEIM_mat[M_DEIM_idx[1:(m-1)], 1:(m-1)] \
    M_DEIM_mat[M_DEIM_idx[1:(m-1)], m]))
    append!(M_DEIM_idx, convert(Int64, argmax(abs.(res))[1]))
    if abs(det(M_DEIM_mat[M_DEIM_idx[1:m], 1:m])) ≤ 1e-80
      n_new = m
      break
    end
  end
  unique!(M_DEIM_idx)
  M_DEIM_err_bound = (Σ[min(n_new + 1,size(Σ)[1])] *
  norm(M_DEIM_mat[M_DEIM_idx,1:n_new] \ I(n_new)))
  M_DEIM_idx = row_idx[M_DEIM_idx]

  sparse_M_DEIM_mat[:,1:n_new], M_DEIM_idx, M_DEIM_err_bound

end =#

function M_DEIM_offline(M_DEIM_mat::Matrix, Σ::Vector)

  (N, n) = size(M_DEIM_mat)
  n_new = n
  M_DEIM_idx = Int64[]
  append!(M_DEIM_idx, convert(Int64, argmax(abs.(M_DEIM_mat[:, 1]))[1]))
  for m in range(2, n)
    res = (M_DEIM_mat[:, m] -
    M_DEIM_mat[:, 1:(m-1)] * (M_DEIM_mat[M_DEIM_idx[1:(m-1)], 1:(m-1)] \
    M_DEIM_mat[M_DEIM_idx[1:(m-1)], m]))
    append!(M_DEIM_idx, convert(Int64, argmax(abs.(res))[1]))
    if abs(det(M_DEIM_mat[M_DEIM_idx[1:m], 1:m])) ≤ 1e-80
      n_new = m
      break
    end
  end
  unique!(M_DEIM_idx)
  M_DEIM_err_bound = (Σ[min(n_new + 1,size(Σ)[1])] *
  norm(M_DEIM_mat[M_DEIM_idx,1:n_new] \ I(n_new)))

  M_DEIM_mat[:,1:n_new], M_DEIM_idx, M_DEIM_err_bound

end

function MDEIM_offline(FEMSpace::SteadyProblem, RBInfo::Info, var::String)

  @info "Building $(RBInfo.nₛ_MDEIM) snapshots of $var"

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  snaps,row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, μ, var)
  sparse_MDEIM_mat, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(sparse_MDEIM_mat, Σ)
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM_idx, FEMSpace.Nₛᵘ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(union(r_idx, c_idx)))

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ

end

function MDEIM_offline(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  var::String)

  @info "Building $(RBInfo.nₛ_MDEIM) snapshots of $var,
  at each time step. This will take some time."

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  snaps,row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, μ, var)
  MDEIM_mat, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(MDEIM_mat, Σ)
  MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx,:]
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx,row_idx,FEMSpace.Nₛᵘ)
  MDEIM_idx_sparse_space, _ = from_vec_to_mat_idx(MDEIM_idx_sparse,FEMSpace.Nₛᵘ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(MDEIM_idx_sparse_space))

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, row_idx, el

end

function DEIM_offline(FEMSpace::SteadyProblem, RBInfo::Info, var::String)

  @info "Building $(RBInfo.nₛ_DEIM) snapshots of $var"

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  snaps = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)
  sparse_DEIM_mat, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(sparse_DEIM_mat, Σ)
  unique!(DEIM_idx)

  DEIM_mat, DEIM_idx, DEIM_err_bound, Σ

end

function DEIM_offline(FEMSpace::UnsteadyProblem, RBInfo::Info, var::String)

  if RBInfo.functional_M_DEIM
    DEIM_offline_functional(FEMSpace, RBInfo, var)
  else
    DEIM_offline_algebraic(FEMSpace, RBInfo, var)
  end

end

function DEIM_offline_algebraic(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  var::String)

  sparse_DEIM_mat = Matrix{Float64}[]
  @info "Building $(RBInfo.nₛ_DEIM) snapshots of $var at each time step."

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  snaps = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)
  sparse_DEIM_mat, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(sparse_DEIM_mat, Σ)
  unique!(DEIM_idx)

  DEIM_mat, DEIM_idx, DEIM_err_bound, Σ

end

function DEIM_offline_functional(FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  var::String)
  error("Functional DEIM not implemented yet")
end

function M_DEIM_online(Mat_nonaffine, Matᵢ::Matrix, idx::Vector)
  Matᵢ\Matrix(reshape(Mat_nonaffine,:,1)[idx,:])
end

#= function check_MDEIM_error_bound(A, Aₘ, Fₘ, ũ, X = nothing)

  if isnothing(X)
    X_inv_sqrt = I(size(X)[1])
  else
    if isfile(joinpath(RBInfo.paths.FEM_structures_path, "X_inv_sqrt.csv"))
      X_inv_sqrt = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "X_inv_sqrt.csv"))
      X_inv_sqrt = sparse(X_inv_sqrt[:, 1], X_inv_sqrt[:, 2], X_inv_sqrt[:, 3])
    else
      @info "Computing inverse of √X, this may take some time"
      H = cholesky(X)
      L = sparse(H.L)
      X_inv_sqrt = ((Matrix(L)' \ I(size(X)[1]))[invperm(H.p), :])'
      @info "Finished computing inverse of √X"
      save_CSV(X_inv_sqrt, joinpath(RBInfo.paths.FEM_structures_path, "X_inv_sqrt.csv"))
    end
  end

  XAX = Matrix(X_inv_sqrt' * A * X_inv_sqrt)
  β = real(eigs(XAX, nev=1, which=:SM)[1])[1]

  MDEIM_err_bound = 0.
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_err_bound.csv"))
    MDEIM_err_bound = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_err_bound.csv"))
  end
  DEIM_err_bound = 0.
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_err_bound.csv"))
    DEIM_err_bound = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_err_bound.csv"))
  end
  last_term = norm(Fₘ - Aₘ * ũ)

  err_bound = (last_term + MDEIM_err_bound[1] + DEIM_err_bound[1]) / β

  return err_bound

end =#
