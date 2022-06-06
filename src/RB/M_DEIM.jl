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

function MDEIM_offline(FESpace::SteadyProblem, RBInfo::Info, var::String)

  @info "Building $(RBInfo.nₛ_MDEIM) snapshots of $var"

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  snaps,row_idx = get_snaps_MDEIM(FESpace, RBInfo, μ, var)
  sparse_MDEIM_mat, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(sparse_MDEIM_mat, Σ)
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM_idx, FESpace.Nₛᵘ)
  el = find_FE_elements(FESpace.V₀, FESpace.Ω, unique(union(r_idx, c_idx)))

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ

end

function MDEIM_offline(FESpace::UnsteadyProblem, RBInfo::Info, var::String)

  if RBInfo.functional_M_DEIM
    return MDEIM_offline_functional(FESpace, RBInfo, var)
  else RBInfo.functional_M_DEIM
    return MDEIM_offline_algebraic(FESpace, RBInfo, var)
  end

end

function MDEIM_offline_algebraic(
  FESpace::UnsteadyProblem,
  RBInfo::Info,
  var::String)

  @info "Building $(RBInfo.nₛ_MDEIM) snapshots of $var,
  at each time step. This will take some time."

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  snaps,row_idx = get_snaps_MDEIM(FESpace, RBInfo, μ, var)
  MDEIM_mat, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(MDEIM_mat, Σ)
  MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx,:]
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx,row_idx,FESpace.Nₛᵘ)
  MDEIM_idx_sparse_space, _ = from_vec_to_mat_idx(MDEIM_idx_sparse,FESpace.Nₛᵘ)
  el = find_FE_elements(FESpace.V₀, FESpace.Ω, unique(MDEIM_idx_sparse_space))

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, row_idx, el

end

function MDEIM_offline_functional(
  FESpace::UnsteadyProblem,
  RBInfo::Info,
  var::String)

  @info "Building $(RBInfo.nₛ_MDEIM) snapshots of $var, at each time step.
  This will take some time."

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  Nₜ = convert(Int64, RBInfo.T / RBInfo.δt)
  δtθ = RBInfo.δt*RBInfo.θ
  times_θ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+δtθ

  ξₖ = get_cell_map(FESpace.Ω)
  Qₕ_cell_point = get_cell_points(FESpace.Qₕ)
  qₖ = get_data(Qₕ_cell_point)
  phys_quadp = lazy_map(evaluate,ξₖ,qₖ)
  ncells = length(phys_quadp)
  nquad_cell = length(phys_quadp[1])
  nquad = nquad_cell*ncells

  refFE_quad = ReferenceFE(lagrangian_quad, Float64, FEMInfo.order)
  V₀_quad = TestFESpace(model, refFE_quad, conformity=:L2)

  for k = 1:RBInfo.nₛ_MDEIM
    @info "Considering Parameter number $k/$(RBInfo.nₛ_MDEIM)"

    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    Param = get_Parametric_specifics(problem_ntuple, RBInfo, μₖ)
    #= if var == "A"
      snapsₖ = [Param.α(phys_quadp[n][q],t_θ)
      for n = 1:ncells for q = 1:nquad_cell for t_θ = times_θ]
    elseif var == "M"
      snapsₖ = [Param.m(phys_quadp[n][q],t_θ)
      for n = 1:ncells for q = 1:nquad_cell for t_θ = times_θ]
    else
      @error "Run MDEIM on A or M only"
    end =#
    snapsₖ = [Param.α(phys_quadp[n][q],t_θ)
    for n = 1:ncells for q = 1:nquad_cell for t_θ = times_θ]
    snapsₖ = reshape(snapsₖ,nquad,Nₜ)
    compressed_snapsₖ, _ = POD(snapsₖ, RBInfo.ϵₛ)
    if k == 1
      global compressed_snaps = compressed_snapsₖ
    else
      global compressed_snaps = hcat(compressed_snaps, compressed_snapsₖ)
    end

  end

  Θmat, Σ = POD(compressed_snaps, RBInfo.ϵₛ)
  Q = size(Param_mat)[2]

  for q = 1:Q
    Θq = FEFunction(V₀_quad,Θmat[:,q])
    if var == "A"
      Matq = (assemble_matrix(∫(∇(FESpace.ϕᵥ)⋅(Θq*∇(FESpace.ϕᵤ(0.0))))*FESpace.dΩ,
       FESpace.V(0.0), FESpace.V₀))
    elseif var == "M"
      Matq = assemble_matrix(∫(FESpace.ϕᵥ*(Θq*FESpace.ϕᵤ(0.0)))*FESpace.dΩ,
      FESpace.V(0.0), FESpace.V₀)
    end
    row_idx, val = findnz(Matq[:])
    if q == 1
      global affine_mat = sparse(row_idx, ones(length(row_idx)),
      val, size(Matq)[1]^2, Q)
    else
      global affine_mat[:,q] = sparse(row_idx, ones(length(row_idx)), val)
    end
  end

  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(affine_mat, Σ)

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ

end

function DEIM_offline(FESpace::SteadyProblem, RBInfo::Info, var::String)

  @info "Building $(RBInfo.nₛ_DEIM) snapshots of $var"

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  snaps = get_snaps_DEIM(FESpace, RBInfo, μ, var)
  sparse_DEIM_mat, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(sparse_DEIM_mat, Σ)
  unique!(DEIM_idx)

  DEIM_mat, DEIM_idx, DEIM_err_bound, Σ

end

function DEIM_offline(FESpace::UnsteadyProblem, RBInfo::Info, var::String)

  if RBInfo.functional_M_DEIM
    DEIM_offline_functional(FESpace, RBInfo, var)
  else
    DEIM_offline_algebraic(FESpace, RBInfo, var)
  end

end

function DEIM_offline_algebraic(
  FESpace::UnsteadyProblem,
  RBInfo::Info,
  var::String)

  sparse_DEIM_mat = Matrix{Float64}[]
  @info "Building $(RBInfo.nₛ_DEIM) snapshots of $var at each time step."

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  snaps = get_snaps_DEIM(FESpace, RBInfo, μ, var)
  sparse_DEIM_mat, Σ = M_DEIM_POD(snaps, RBInfo.ϵₛ)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(sparse_DEIM_mat, Σ)
  unique!(DEIM_idx)

  DEIM_mat, DEIM_idx, DEIM_err_bound, Σ

end

function DEIM_offline_functional(FESpace::UnsteadyProblem,
  RBInfo::Info,
  var::String)
  @abstractmethod
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
