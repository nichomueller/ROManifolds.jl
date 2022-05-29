include("../FEM/FEM_utils.jl")
include("M_DEIM_build_snapshots.jl")

function M_DEIM_POD(S::SparseMatrixCSC, ϵ = 1e-5)

  S̃ = copy(S)
  Nₕ = size(S̃)[1]
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

  while N ≤ size(S̃)[2]-2 && (M_DEIM_err_bound > ϵ || cumulative_energy / total_energy < 1.0 - ϵ ^ 2)

    N += 1
    cumulative_energy += Σ[N] ^ 2
    M_DEIM_err_bound = Σ[N + 1] * mult_factor

    @info "(M)DEIM-POD loop number $N, projection error = $M_DEIM_err_bound"

  end
  M_DEIM_mat = M_DEIM_mat[:, 1:N]
  _, col_idx, val = findnz(sparse(M_DEIM_mat))
  sparse_M_DEIM_mat = sparse(repeat(row_idx, N), col_idx, val, Nₕ, N)

  @info "Basis number obtained via POD is $N, projection error ≤ $M_DEIM_err_bound"

  return sparse_M_DEIM_mat, Σ

end

function M_DEIM_POD(S::Matrix, ϵ = 1e-5)

  S̃ = copy(S)
  C = S̃'*S̃
  V, Σ, _ = svd(C)
  Σ = sqrt.(Σ)

  total_energy = sum(Σ .^ 2)
  cumulative_energy = 0.0
  N = 0

  while N ≤ size(S̃)[2]-1 && cumulative_energy / total_energy < 1.0 - ϵ ^ 2
    N += 1
    cumulative_energy += Σ[N] ^ 2
    @info "POD loop number $N, cumulative energy = $cumulative_energy"
  end

  V = V[:, 1:N]
  M_DEIM_mat = S̃*V
  for n = 1:N
    M_DEIM_mat[:,n] /= Σ[n]
  end

  @info "Basis number obtained via POD is $N, projection error ≤ $((sqrt(abs(1 - cumulative_energy / total_energy))))"

  return M_DEIM_mat, Σ

end

#= function M_DEIM_POD_functional(S::Array, ϵ=1e-5)

  S̃ = copy(S)
  M_DEIM_mat, Σ, _ = svd(S̃)
  (r,c) = size(M_DEIM_mat)

  if r ≤ c

    M_DEIM_mat = M_DEIM_mat[:, 1:r]

  else

    total_energy = sum(Σ .^ 2)
    cumulative_energy = 0.0
    N = 0

    while N ≤ size(S̃)[2]-1 && cumulative_energy / total_energy < 1.0 - ϵ ^ 2

      N += 1
      cumulative_energy += Σ[N] ^ 2

      @info "(M)DEIM-POD loop number $N, projection error = $((sqrt(abs(1 - cumulative_energy / total_energy))))"

    end
    M_DEIM_mat = M_DEIM_mat[:, 1:N]

  end

  return M_DEIM_mat,  Σ

end =#

function M_DEIM_offline(sparse_M_DEIM_mat::SparseMatrixCSC, Σ::Vector)

  row_idx, _, _ = findnz(sparse_M_DEIM_mat)
  unique!(row_idx)
  M_DEIM_mat = Matrix(sparse_M_DEIM_mat[row_idx, :])

  (N, n) = size(M_DEIM_mat)
  n_new = n
  M_DEIM_idx = Int64[]
  append!(M_DEIM_idx, convert(Int64, argmax(abs.(M_DEIM_mat[:, 1]))[1]))
  for m in range(2, n)
    res = M_DEIM_mat[:, m] - M_DEIM_mat[:, 1:(m-1)] * (M_DEIM_mat[M_DEIM_idx[1:(m-1)], 1:(m-1)] \ M_DEIM_mat[M_DEIM_idx[1:(m-1)], m])
    append!(M_DEIM_idx, convert(Int64, argmax(abs.(res))[1]))
    if abs(det(M_DEIM_mat[M_DEIM_idx[1:m], 1:m])) ≤ 1e-80
      n_new = m
      break
    end
  end
  unique!(M_DEIM_idx)
  M_DEIM_err_bound = Σ[min(n_new + 1,size(Σ)[1])] * norm(M_DEIM_mat[M_DEIM_idx,1:n_new] \ I(n_new))
  M_DEIM_idx = row_idx[M_DEIM_idx]

  sparse_M_DEIM_mat[:,1:n_new], M_DEIM_idx, M_DEIM_err_bound

end

function M_DEIM_offline(M_DEIM_mat::Matrix, Σ::Vector)

  (N, n) = size(M_DEIM_mat)
  n_new = n
  M_DEIM_idx = Int64[]
  append!(M_DEIM_idx, convert(Int64, argmax(abs.(M_DEIM_mat[:, 1]))[1]))
  for m in range(2, n)
    res = M_DEIM_mat[:, m] - M_DEIM_mat[:, 1:(m-1)] * (M_DEIM_mat[M_DEIM_idx[1:(m-1)], 1:(m-1)] \ M_DEIM_mat[M_DEIM_idx[1:(m-1)], m])
    append!(M_DEIM_idx, convert(Int64, argmax(abs.(res))[1]))
    if abs(det(M_DEIM_mat[M_DEIM_idx[1:m], 1:m])) ≤ 1e-80
      n_new = m
      break
    end
  end
  unique!(M_DEIM_idx)
  M_DEIM_err_bound = Σ[min(n_new + 1,size(Σ)[1])] * norm(M_DEIM_mat[M_DEIM_idx,1:n_new] \ I(n_new))

  M_DEIM_mat[:,1:n_new], M_DEIM_idx, M_DEIM_err_bound

end

function M_DEIM_online(mat_nonaffine, M_DEIMᵢ_mat, M_DEIM_idx)

  vec_nonaffine = Matrix(reshape(mat_nonaffine,:,1)[M_DEIM_idx,:])
  M_DEIM_coeffs = M_DEIMᵢ_mat \ vec_nonaffine

  M_DEIM_coeffs

end

function MDEIM_offline(FE_space::SteadyProblem, ROM_info::Info, var::String)

  @info "Building $(ROM_info.nₛ_MDEIM) snapshots of $var"

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))

  if var === "A"
    snaps = build_A_snapshots(FE_space, ROM_info, μ)
  elseif var === "M"
    snaps = build_M_snapshots(FE_space, ROM_info, μ)
  else
    @error "Run MDEIM on A or M only"
  end

  sparse_MDEIM_mat, Σ = M_DEIM_POD(snaps, ROM_info.ϵₛ)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(sparse_MDEIM_mat, Σ)

  r_idx, c_idx = from_vec_to_mat_idx(MDEIM_idx, FE_space.Nₛᵘ)

  el = find_FE_elements(FE_space.V₀, FE_space.Ω, unique(union(r_idx, c_idx)))

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ

end

function MDEIM_offline(FE_space::UnsteadyProblem, ROM_info::Info, var::String)

  if ROM_info.space_time_M_DEIM
    return MDEIM_offline_spacetime(FE_space, ROM_info, var)
  elseif ROM_info.functional_M_DEIM
    return MDEIM_offline_functional(FE_space, ROM_info, var)
  else
    return MDEIM_offline_standard(FE_space, ROM_info, var)
  end

end

function MDEIM_offline_standard(FE_space::UnsteadyProblem, ROM_info::Info, var::String)

  @info "Building $(ROM_info.nₛ_MDEIM) snapshots of $var, at each time step. This will take some time."

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))

  for k = 1:ROM_info.nₛ_MDEIM
    @info "Considering parameter number $k, need $(ROM_info.nₛ_MDEIM-k) more!"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    if var === "A"
      snapsₖ = build_A_snapshots(FE_space, ROM_info, μₖ)
    elseif var === "M"
      snapsₖ = build_M_snapshots(FE_space, ROM_info, μₖ)
    else
      @error "Run MDEIM on A or M only"
    end
    compressed_snapsₖ, _ = M_DEIM_POD(snapsₖ, ROM_info.ϵₛ)
    if k === 1
      global compressed_snaps = compressed_snapsₖ
    else
      global compressed_snaps = hcat(compressed_snaps, compressed_snapsₖ)
    end
  end

  sparse_MDEIM_mat, Σ = M_DEIM_POD(compressed_snaps, ROM_info.ϵₛ)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(sparse_MDEIM_mat, Σ)

  r_idx, c_idx = from_vec_to_mat_idx(MDEIM_idx, FE_space.Nₛᵘ)

  el = find_FE_elements(FE_space.V₀, FE_space.Ω, unique(union(r_idx, c_idx)))

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ

end

function MDEIM_offline_spacetime(FE_space::UnsteadyProblem, ROM_info::Info, var::String)

  @info "Building at each time step $(ROM_info.nₛ_MDEIM) snapshots of $var. This will take some time."

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  for (nₜ,t) = enumerate(times_θ)
    @info "Considering time step $nₜ/$Nₜ"
    if var === "A"
      snapsₜ, row_idx = build_A_snapshots(FE_space, ROM_info, μ, t)
    elseif var === "M"
      snapsₜ, row_idx = build_M_snapshots(FE_space, ROM_info, μ, t)
    else
      @error "Run MDEIM on A or M only"
    end
    if nₜ === 1
      global row_idx = row_idx
    end
    compressed_snapsₜ,_ = M_DEIM_POD(snapsₜ, ROM_info.ϵₛ)
    if nₜ === 1
      global compressed_snaps = compressed_snapsₜ
    else
      if size(compressed_snaps)[2] != size(compressed_snapsₜ)[2]
        c = size(compressed_snaps)[2]
        cₜ = size(compressed_snapsₜ)[2]
        if c > cₜ
          basis_snapsₜ, _, _ = svd(compressed_snapsₜ)
          compressed_snapsₜ = basis_snapsₜ[:,1:c]
        else
          compressed_snapsₜ = compressed_snapsₜ[:,1:c]
        end
      end
      global compressed_snaps = vcat(compressed_snaps, compressed_snapsₜ)
    end

  end

  MDEIM_mat_init, Σ = M_DEIM_POD(compressed_snaps, ROM_info.ϵₛ)
  MDEIM_mat_tmp, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(MDEIM_mat_init, Σ)
  MDEIM_idx_space, MDEIM_idx_time = from_spacetime_to_space_time_idx_vec(MDEIM_idx, length(row_idx))
  unique!(MDEIM_idx_time)

  MDEIM_mat = zeros(length(row_idx), length(MDEIM_idx)*length(MDEIM_idx_time))
  for i=1:length(MDEIM_idx_time)
    MDEIM_mat[:,(i-1)*length(MDEIM_idx)+1:i*length(MDEIM_idx)] = MDEIM_mat_tmp[(MDEIM_idx_time[i]-1)*length(row_idx)+1:MDEIM_idx_time[i]*length(row_idx),:]
  end

  idx_sparse = from_full_idx_to_sparse_idx(row_idx,FE_space.Nₛᵘ,Nₜ)
  MDEIM_idx_sparse = idx_sparse[MDEIM_idx]
  MDEIM_idx_space_sparse, _ = from_spacetime_to_space_time_idx_mat(MDEIM_idx_sparse, FE_space.Nₛᵘ)
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM_idx_space_sparse, FE_space.Nₛᵘ)
  el = find_FE_elements(FE_space.V₀, FE_space.Ω, unique(union(r_idx, c_idx)))

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ, row_idx

end

function MDEIM_offline_functional(FE_space::UnsteadyProblem, ROM_info::Info, var::String)

  include("../FEM/LagrangianQuadRefFEs.jl")
  @info "Building $(ROM_info.nₛ_MDEIM) snapshots of $var, at each time step. This will take some time."

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ
  param = get_parametric_specifics(ROM_info, parse.(Float64, split(chop(μ[1]; head=1, tail=1), ',')))
  Qₕ_cell_data = get_data(FE_space.Qₕ)
  qₕ = Qₕ_cell_data[rand(1:num_cells(FE_space.Ω))]
  xₕ = get_coordinates(qₕ)
  nquad = length(xₕ)

  for k = 1:ROM_info.nₛ_MDEIM
    @info "Considering parameter number $k, need $(ROM_info.nₛ_MDEIM-k) more!"

    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    param = get_parametric_specifics(ROM_info, μₖ)
    snapsₖ = [param.α(xₕ[n],t_θ) for n = 1:nquad for t_θ = times_θ]
    #= if var === "A"
      snapsₖ = [param.α(xₕ[n],t_θ) for n = 1:nquad for t_θ = times_θ]
    elseif var === "M"
      snapsₖ = [param.m(xₕ[n],t_θ) for n = 1:nquad for t_θ = times_θ]
    else
      @error "Run MDEIM on A or M only"
    end =#
    snapsₖ = reshape(snapsₖ,nquad,Nₜ)
    compressed_snapsₖ, _ = M_DEIM_POD(snapsₖ, ROM_info.ϵₛ)
    if k === 1
      global compressed_snaps = compressed_snapsₖ
    else
      global compressed_snaps = hcat(compressed_snaps, compressed_snapsₖ)
    end

  end

  param_mat, Σ = M_DEIM_POD(compressed_snaps, ROM_info.ϵₛ)
  param_mat_rep = repeat(param_mat, outer=[num_cells(FE_space.Ω),1])

  refFE_quad = ReferenceFE(lagrangian_quad, Float64, problem_info.order, FE_space.Ω)
  V_quad =  TrialFESpace(TestFESpace(param.model, refFE_quad, conformity=:L2))

  for q = 1:size(param_mat)[2]
    f_q = FEFunction(V_quad,param_mat_rep[:,q])
    Matq = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (f_q * ∇(FE_space.ϕᵤ(0.0)))) * FE_space.dΩ, FE_space.V(0.0), FE_space.V₀)
    row_idx, val = findnz(Matq[:])
    if q === 1
      global affine_mat = sparse(row_idx, ones(length(row_idx)), val, size(Matq)[1]^2, size(param_mat)[2])
    else
      global affine_mat[:,q] = sparse(row_idx, ones(length(row_idx)), val)
    end
  end

  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(affine_mat, Σ)

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ

end

function DEIM_offline(FE_space::SteadyProblem, ROM_info::Info, var::String)

  @info "Building $(ROM_info.nₛ_DEIM) snapshots of $var"

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))

  if var === "F"
    snaps = build_F_snapshots(FE_space, ROM_info, μ)
  elseif var === "H"
    snaps = build_H_snapshots(FE_space, ROM_info, μ)
  else
    @error "Run DEIM on F or H only"
  end

  sparse_DEIM_mat, Σ = M_DEIM_POD(snaps, ROM_info.ϵₛ)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(sparse_DEIM_mat, Σ)
  unique!(DEIM_idx)

  DEIM_mat, DEIM_idx, DEIM_err_bound, Σ

end

function DEIM_offline(FE_space::UnsteadyProblem, ROM_info::Info, var::String)

  if ROM_info.space_time_M_DEIM
    DEIM_offline_spacetime(FE_space, ROM_info, var)
  else
    DEIM_offline_standard(FE_space, ROM_info, var)
  end

end

function DEIM_offline_standard(FE_space::UnsteadyProblem, ROM_info::Info, var::String)

  sparse_DEIM_mat = Matrix{Float64}[]
  @info "Building $(ROM_info.nₛ_DEIM) snapshots of $var at each time step."

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))

  for k = 1:ROM_info.nₛ_MDEIM
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    if var === "F"
      snapsₖ = build_F_snapshots(FE_space, ROM_info, μₖ)
    elseif var === "H"
      snapsₖ = build_H_snapshots(FE_space, ROM_info, μₖ)
    else
      @error "Run DEIM on F or H only"
    end

    compressed_snapsₖ, _ = M_DEIM_POD(snapsₖ, ROM_info.ϵₛ)
    if k === 1
      global compressed_snaps = compressed_snapsₖ
    else
      global compressed_snaps = hcat(compressed_snaps,compressed_snapsₖ)
    end
    global Σ = Σ
  end

  sparse_DEIM_mat, Σ = M_DEIM_POD(compressed_snaps, ROM_info.ϵₛ)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(sparse_DEIM_mat, Σ)
  unique!(DEIM_idx)

  DEIM_mat, DEIM_idx, DEIM_err_bound, Σ

end

function DEIM_offline_spacetime(FE_space::UnsteadyProblem, ROM_info::Info, var::String)

  @info "Building at each time step $(ROM_info.nₛ_MDEIM) snapshots of $var."

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)

  for nₜ = 1:Nₜ
    @info "Considering time step $nₜ/$Nₜ"
    if var === "F"
      snapsₜ = build_F_snapshots(FE_space, ROM_info, μ, nₜ)
    elseif var === "H"
      snapsₜ = build_H_snapshots(FE_space, ROM_info, μ, nₜ)
    else
      @error "Run DEIM on F or H only"
    end
    compressed_snapsₜ, _ = M_DEIM_POD(snapsₜ, ROM_info.ϵₛ)
    if nₜ === 1
      global compressed_snaps = compressed_snapsₜ
    else
      if size(compressed_snaps)[2] != size(compressed_snapsₜ)[2]
        c = size(compressed_snaps)[2]
        cₜ = size(compressed_snapsₜ)[2]
        if c > cₜ
          basis_snapsₜ, _, _ = svd(compressed_snapsₜ)
          compressed_snapsₜ = basis_snapsₜ[:,1:c]
        else
          compressed_snapsₜ = compressed_snapsₜ[:,1:c]
        end
      end
      global compressed_snaps = vcat(compressed_snaps, compressed_snapsₜ)
    end
  end

  sparse_DEIM_mat, Σ = M_DEIM_POD(compressed_snaps, ROM_info.ϵₛ)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(sparse_DEIM_mat, Σ)
  unique!(DEIM_idx)
  DEIM_mat = reshape(DEIM_mat, FE_space.Nₛᵘ, :)

  DEIM_mat, DEIM_idx, DEIM_err_bound, Σ

end

function check_MDEIM_error_bound(A, Aₘ, Fₘ, ũ, X = nothing)

  if isnothing(X)
    X_inv_sqrt = I(size(X)[1])
  else
    if isfile(joinpath(ROM_info.paths.FEM_structures_path, "X_inv_sqrt.csv"))
      X_inv_sqrt = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "X_inv_sqrt.csv"))
      X_inv_sqrt = sparse(X_inv_sqrt[:, 1], X_inv_sqrt[:, 2], X_inv_sqrt[:, 3])
    else
      @info "Computing inverse of √X, this may take some time"
      H = cholesky(X)
      L = sparse(H.L)
      X_inv_sqrt = ((Matrix(L)' \ I(size(X)[1]))[invperm(H.p), :])'
      @info "Finished computing inverse of √X"
      save_CSV(X_inv_sqrt, joinpath(ROM_info.paths.FEM_structures_path, "X_inv_sqrt.csv"))
    end
  end

  XAX = Matrix(X_inv_sqrt' * A * X_inv_sqrt)
  β = real(eigs(XAX, nev=1, which=:SM)[1])[1]

  MDEIM_err_bound = 0.
  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_err_bound.csv"))
    MDEIM_err_bound = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_err_bound.csv"))
  end
  DEIM_err_bound = 0.
  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_err_bound.csv"))
    DEIM_err_bound = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_err_bound.csv"))
  end
  last_term = norm(Fₘ - Aₘ * ũ)

  err_bound = (last_term + MDEIM_err_bound[1] + DEIM_err_bound[1]) / β

  return err_bound

end
