include("MV_snapshots.jl")

function M_DEIM_POD(S::Matrix, ϵ = 1e-5)

  S̃ = copy(S)
  M_DEIM_mat, Σ, _ = svd(S̃)

  total_energy = sum(Σ .^ 2)
  cumulative_energy = 0.0
  N = 0
  M_DEIM_err_bound = 1e5
  crit_val = norm(inv(M_DEIM_mat'M_DEIM_mat))
  mult_factor = sqrt(size(S̃)[2]) * crit_val

  while N < size(S̃)[2] && (M_DEIM_err_bound > ϵ ||
    cumulative_energy / total_energy < 1.0 - ϵ ^ 2)
    N += 1
    cumulative_energy += Σ[N] ^ 2
    if N==size(S̃)[2]
      M_DEIM_err_bound = 0.
    else
      M_DEIM_err_bound = Σ[N+1]*mult_factor
    end
    @info "(M)DEIM-POD loop number $N, projection error = $M_DEIM_err_bound"
  end

  @info "Basis number obtained via POD is $N, projection error ≤ $M_DEIM_err_bound"

  return M_DEIM_mat[:,1:N], Σ

end

function M_DEIM_offline(M_DEIM_mat::Matrix, Σ::Vector)

  (N, n) = size(M_DEIM_mat)
  n_new = n
  M_DEIM_idx = Int64[]
  append!(M_DEIM_idx, Int(argmax(abs.(M_DEIM_mat[:, 1]))[1]))
  for m = 2:n
    res = (M_DEIM_mat[:, m] -
    M_DEIM_mat[:, 1:m-1] * (M_DEIM_mat[M_DEIM_idx[1:m-1], 1:m-1] \
    M_DEIM_mat[M_DEIM_idx[1:m-1], m]))
    append!(M_DEIM_idx, convert(Int64, argmax(abs.(res))[1]))
    if abs(det(M_DEIM_mat[M_DEIM_idx[1:m], 1:m])) ≤ 1e-80
      #= error("Something went wrong with the construction of (M)DEIM basis:
        obtaining singular nested matrices during (M)DEIM offline phase") =#
      n_new = m
      break
    end
  end
  unique!(M_DEIM_idx)
  M_DEIM_err_bound = (Σ[min(n_new + 1,length(Σ))] *
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
  timesθ = get_timesθ(RBInfo)
  MDEIM_mat,row_idx,Σ = get_snaps_MDEIM(FEMSpace,RBInfo,μ,var)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(MDEIM_mat, Σ)
  MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx,:]
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx,row_idx,FEMSpace.Nₛᵘ)
  MDEIM_idx_sparse_space, _ = from_vec_to_mat_idx(MDEIM_idx_sparse,FEMSpace.Nₛᵘ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(MDEIM_idx_sparse_space))

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, row_idx, el

end

function modify_timesθ_and_MDEIM_idx(
  MDEIM_idx::Vector,
  RBInfo::Info,
  RBVars::PoissonUnsteady) ::Tuple
  timesθ = get_timesθ(RBInfo)
  idx_space, idx_time = from_vec_to_mat_idx(MDEIM_idx,RBVars.S.Nₛᵘ^2)
  idx_time_mod = label_sorted_elems(idx_time)
  timesθ_mod = timesθ[unique(sort(idx_time))]
  MDEIM_idx_mod = (idx_time_mod.-1)*RBVars.S.Nₛᵘ^2+idx_space
  timesθ_mod,MDEIM_idx_mod
end

function DEIM_offline(
  FEMSpace::SteadyProblem,
  RBInfo::Info,
  var::String) ::Tuple

  @info "Building $(RBInfo.nₛ_DEIM) snapshots of $var"

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  snaps = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)
  sparse_DEIM_mat, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(sparse_DEIM_mat, Σ)
  unique!(DEIM_idx)

  DEIM_mat, DEIM_idx, DEIM_err_bound, Σ

end

function DEIM_offline(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  var::String) ::Tuple

  @info "Building $(RBInfo.nₛ_DEIM) snapshots of $var, at each time step."

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  timesθ = get_timesθ(RBInfo)
  DEIM_mat,Σ = get_snaps_DEIM(FEMSpace,RBInfo,μ,var)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(DEIM_mat, Σ)
  DEIMᵢ_mat = DEIM_mat[DEIM_idx,:]

  DEIM_mat, DEIM_idx, DEIMᵢ_mat

end

function M_DEIM_online(Mat_nonaffine, Matᵢ::Matrix, idx::Vector)
  Matᵢ\Matrix(reshape(Mat_nonaffine,:,1)[idx,:])
end
