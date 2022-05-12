include("../FEM/FEM_utils.jl")
include("M_DEIM_build_snapshots.jl")

function M_DEIM_POD(S, ϵ = 1e-5)

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

  return sparse_M_DEIM_mat,  Σ

end

function M_DEIM_offline(sparse_M_DEIM_mat, Σ)

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

function M_DEIM_online(mat_nonaffine, M_DEIMᵢ_mat, M_DEIM_idx)

  vec_nonaffine = Matrix(reshape(mat_nonaffine,:,1)[M_DEIM_idx,:])
  M_DEIM_coeffs = M_DEIMᵢ_mat \ vec_nonaffine

  M_DEIM_coeffs

end

function MDEIM_offline(problem_info::ProblemSpecifics, ROM_info, var::String)

  if var === "A"
    snaps = build_A_snapshots(problem_info, ROM_info)
  elseif var === "M"
    snaps = build_M_snapshots(problem_info, ROM_info)
  else
    @error "Run MDEIM on A or M only"
  end

  sparse_MDEIM_mat, Σ = M_DEIM_POD(snaps, ROM_info.ϵₛ)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(sparse_MDEIM_mat, Σ)

  Nₕ = convert(Int64, sqrt(size(snaps)[1]))
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM_idx, Nₕ)

  el = find_FE_elements(problem_info, ROM_info, unique(union(r_idx, c_idx)))

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ

end

function MDEIM_offline(problem_info::ProblemSpecificsUnsteady, ROM_info, var::String)

  if ROM_info.space_time_M_DEIM
    return MDEIM_offline_spacetime(problem_info, ROM_info, var)
  else
    return MDEIM_offline_standard(problem_info, ROM_info, var)
  end

end

function MDEIM_offline_standard(problem_info::ProblemSpecificsUnsteady, ROM_info, var::String)

  @info "Building $(ROM_info.nₛ_MDEIM) snapshots of $var, at each time step. This will take some time."

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))

  for k = 1:ROM_info.nₛ_MDEIM
    @info "Considering parameter number $k, need $(ROM_info.nₛ_MDEIM-k) more!"
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    if var === "A"
      snapsₖ = build_A_snapshots(problem_info, ROM_info, μₖ)
    elseif var === "M"
      snapsₖ = build_M_snapshots(problem_info, ROM_info, μₖ)
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

  Nₕ = convert(Int64, sqrt(size(MDEIM_mat)[1]))
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM_idx, Nₕ)

  el = find_FE_elements(problem_info, ROM_info, unique(union(r_idx, c_idx)))

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ

end

function MDEIM_offline_spacetime(problem_info::ProblemSpecificsUnsteady, ROM_info, var::String)

  @info "Building at each time step $(ROM_info.nₛ_MDEIM) snapshots of $var. This will take some time."

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)
  δtθ = ROM_info.δt*ROM_info.θ
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ

  for (nₜ,t) = enumerate(times_θ)
    @info "Considering time step $nₜ/$Nₜ"
    if var === "A"
      snapsₜ = build_A_snapshots(problem_info, ROM_info, μ, t)
    elseif var === "M"
      snapsₜ = build_M_snapshots(problem_info, ROM_info, μ, t)
    else
      @error "Run MDEIM on A or M only"
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

  sparse_MDEIM_mat, Σ = M_DEIM_POD(compressed_snaps, ROM_info.ϵₛ)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(sparse_MDEIM_mat, Σ)

  Nₕ = convert(Int64, sqrt(size(MDEIM_mat)[1]/Nₜ))
  MDEIM_idx_space, MDEIM_idx_time = from_spacetime_to_space_time_idx_mat(MDEIM_idx, Nₕ)
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM_idx_space, Nₕ)

  el = find_FE_elements(problem_info, ROM_info, unique(union(r_idx, c_idx)))
  MDEIM_mat = reshape(MDEIM_mat, Nₕ^2, :)[:,MDEIM_idx_time[:]]

  MDEIM_mat, MDEIM_idx, el, MDEIM_err_bound, Σ

end



function DEIM_offline(problem_info::ProblemSpecifics, ROM_info, var::String)

  if var === "F"
    snaps = build_F_snapshots(problem_info, ROM_info)
  elseif var === "H"
    snaps = build_H_snapshots(problem_info, ROM_info)
  else
    @error "Run DEIM on F or H only"
  end

  sparse_DEIM_mat, Σ = M_DEIM_POD(snaps, ROM_info.ϵₛ)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(sparse_DEIM_mat, Σ)
  unique!(DEIM_idx)

  DEIM_mat, DEIM_idx, DEIM_err_bound, Σ

end

function DEIM_offline(problem_info::ProblemSpecificsUnsteady, ROM_info, var::String)

  if ROM_info.space_time_M_DEIM
    DEIM_offline_spacetime(problem_info, ROM_info, var)
  else
    DEIM_offline_standard(problem_info, ROM_info, var)
  end

end

function DEIM_offline_standard(problem_info::ProblemSpecificsUnsteady, ROM_info, var::String)

  sparse_DEIM_mat = Matrix{Float64}[]
  @info "Building $(ROM_info.nₛ_DEIM) snapshots of $var at each time step."

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))

  for k = 1:ROM_info.nₛ_MDEIM
    μₖ = parse.(Float64, split(chop(μ[k]; head=1, tail=1), ','))
    if var === "F"
      snapsₖ = build_F_snapshots(problem_info, ROM_info, μₖ)
    elseif var === "H"
      snapsₖ = build_H_snapshots(problem_info, ROM_info, μₖ)
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

function DEIM_offline_spacetime(problem_info::ProblemSpecificsUnsteady, ROM_info, var::String)

  @info "Building at each time step $(ROM_info.nₛ_MDEIM) snapshots of $var."

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)

  for nₜ = 1:Nₜ
    @info "Considering time step $nₜ/$Nₜ"
    if var === "F"
      snapsₜ = build_F_snapshots(problem_info, ROM_info, μ, nₜ)
    elseif var === "H"
      snapsₜ = build_H_snapshots(problem_info, ROM_info, μ, nₜ)
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
  Nₕ = convert(Int64, size(DEIM_mat)[1]/Nₜ)
  DEIM_mat = reshape(DEIM_mat, Nₕ, :)

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
