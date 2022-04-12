include("../utils/general.jl")
include("../FEM/FEM_utils.jl")

function POD(S, ϵ = 1e-5, X = nothing; check_Σ = false)
  #=MODIFY
  =#

  S̃ = copy(S)

  if !isnothing(X)
    if !issparse(X)
      X = sparse(X)
    end

    H = cholesky(X)
    L = sparse(H.L)
    mul!(S̃, L', S̃[H.p, :])
  end

  if issparse(S̃)
    U, Σ, _ = svds(S̃; nsv=size(S̃)[2] - 1)[1]
  else
    U, Σ, _ = svd(S̃)
  end

  total_energy = sum(Σ .^ 2)
  cumulative_energy = 0.0
  N = 0

  if !check_Σ
    while cumulative_energy / total_energy < 1.0 - ϵ ^ 2 && N < size(S̃)[2]
      N += 1
      cumulative_energy += Σ[N] ^ 2
      @info "POD loop number $N, cumulative energy = $cumulative_energy"
    end
  else
    while N < size(S̃)[2] && (Σ[N + 1] > ϵ || cumulative_energy / total_energy < 1.0 - ϵ ^ 2)
      N += 1
      cumulative_energy += Σ[N] ^ 2
      @info "POD loop number $N, cumulative energy = $cumulative_energy"
    end
  end

  @info "Basis number obtained via POD is $N, projection error ≤ $((sqrt(abs(1 - cumulative_energy / total_energy))))"

  if issparse(U)
    U = Matrix(U)
  end

  if !isnothing(X)
    return Matrix((L' \ U[:, 1:N])[invperm(H.p), :]), Σ #Matrix((L' \ U[:, 1:N]))
  else
    return U[:, 1:N], Σ
  end

end

function rPOD(S, ϵ=1e-5, X=nothing, q=1, m=nothing)
  #=MODIFY
  =#

  if isnothing(m)
    m = size(S)[2] ./ 2
  end

  Ω = randn!(size(S)[1], m)
  (SS, SΩ, Y, B) = (zeros(size(S)[1], size(S)[1]), zeros(size(S)[1], size(Ω)[2]), similar(Ω), zeros(size(Ω)[2], size(S)[2]))
  mul!(Y, mul!(SS, S, S')^q, mul!(SΩ, S, Ω))
  (Q, R) = qr!(Y)
  mul!(B, Q', S)

  if !(X === nothing)
    if issparse(X) === false
      X = sparse(X)
    end
    H = cholesky(X)
    L = sparse(H.L)
    mul!(B, L', B[H.p, :])

  end

  U, Σ, _ = svd(B)

  total_energy = sum(Σ .^ 2)
  cumulative_energy = 0.0
  N = 0

  while cumulative_energy / total_energy < 1.0 - ϵ ^ 2 && N < size(B)[2]
    N += 1
    cumulative_energy += Σ[N] ^ 2
    @info "POD loop number $N, cumulative energy = $cumulative_energy"
  end

  @info "Basis number obtained via POD is $N, projection error <= $(abs((sqrt(1 - cumulative_energy / total_energy))))"
  V = Q * U[:, N]

  if !isnothing(X)
    return Matrix((L' \ V[:, 1:N])[invperm(H.p), :])
  else
    return V
  end

end

function DEIM_offline(S, ϵ = 1e-5)
  #=MODIFY
  =#

  DEIM_mat, Σ = POD(S, ϵ; check_Σ = true)

  (N, n) = size(DEIM_mat)
  DEIM_idx = Int64[]

  append!(DEIM_idx, convert(Int64, argmax(abs.(DEIM_mat[:, 1]))[1]))
  for m in range(2, n)
    res = DEIM_mat[:, m] - DEIM_mat[:, 1:(m-1)] * (DEIM_mat[DEIM_idx[1:(m-1)], 1:(m-1)] \ DEIM_mat[DEIM_idx[1:(m-1)], m])
    append!(DEIM_idx, convert(Int64, argmax(abs.(res))[1]))
  end

  DEIM_err_bound = Σ[n+1] * norm(DEIM_mat[DEIM_idx, :] \ I(n))

  (DEIM_mat, DEIM_idx, DEIM_err_bound, Σ)

end

function DEIM_online(vec_nonaffine, DEIM_mat, DEIM_idx)
  #=MODIFY
  =#

  vec_affine = zeros(size(vec_nonaffine))

  DEIM_coeffs = DEIM_mat[DEIM_idx[:], :] \ vec_nonaffine[DEIM_idx[:]]
  mul!(vec_affine, DEIM_mat, DEIM_coeffs)

  return DEIM_coeffs, vec_affine

end

function MDEIM_online(mat_nonaffine, MDEIM_mat, MDEIM_idx, row_idx = nothing, col_idx = nothing)
  #=MODIFY
  S is already in the correct format, so it is a matrix of size (R*C, quantity), while mat_nonaffine is of size (R, C)
  =#

  if !issparse(mat_nonaffine)
    vec_nonaffine = reshape(mat_nonaffine, :, 1)
  else
    vec_nonaffine = zeros(length(row_idx))
    full_idx = hcat(row_idx, col_idx)
    red_row_idx, red_col_idx, red_val = findnz(mat_nonaffine)
    red_idx = hcat(red_row_idx, red_col_idx)
    q = indexin(collect(eachrow(red_idx)), collect(eachrow(full_idx)))
    vec_nonaffine[q] = red_val
  end
  MDEIM_coeffs = MDEIM_mat[MDEIM_idx[:], :] \ vec_nonaffine[MDEIM_idx[:]]
  vec_affine = MDEIM_mat * MDEIM_coeffs

  if !issparse(mat_nonaffine)
    mat_affine = reshape(vec_affine, size(mat_nonaffine)[1], size(mat_nonaffine)[2])
  else
    mat_affine = sparse(row_idx, col_idx, vec_affine)
  end

  return MDEIM_coeffs, mat_affine

end

function build_A_snapshots(problem_info, ROM_info, nₛ_MDEIM::Int, μ::Array)

  @info "Building $nₛ_MDEIM snapshots of the matrix A"

  row_idx, col_idx, val, A = Int64[], Int64[], zeros(0), zeros(0)
  for i_nₛ = 1:nₛ_MDEIM
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    FE_space = get_FE_space(problem_info, parametric_info.model)
    A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
    row_idx, col_idx, val = findnz(A_i)
    if i_nₛ === 1
      A = zeros(length(row_idx), nₛ_MDEIM)
    end
    A[:, i_nₛ] = val
  end

  A, row_idx, col_idx

end

function find_FE_elements(idx::Array, σₖ::Table)

  el = Int64[]
  for i = 1:length(idx)
    for j = 1:size(σₖ)[1]
      if idx[i] in abs.(σₖ[j])
        append!(el, j)
      end
    end
  end

  el

end

function MDEIM_offline(problem_info, ROM_info, nₛ_MDEIM::Int, μ::Array)

  A_snapshots, row_idx, col_idx = build_A_snapshots(problem_info, ROM_info, nₛ_MDEIM, μ)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound, Σ = DEIM_offline(A_snapshots, ROM_info.ϵₛ)
  idx = unique(union(row_idx[MDEIM_idx], col_idx[MDEIM_idx]))

  μ1 = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  parametric_info = get_parametric_specifics(ROM_info, μ1)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  el = find_FE_elements(idx, FE_space.σₖ)

  MDEIM_mat, MDEIM_idx, row_idx, col_idx, el, MDEIM_err_bound, Σ

end

function build_sparse_LHS(problem_info, ROM_info, μ_i::Array, el::Array)

  parametric_info = get_parametric_specifics(ROM_info, μ_i)
  FE_space = get_FE_space(problem_info, parametric_info.model)

  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * problem_info.order)
  assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (parametric_info.α * ∇(FE_space.ϕᵤ))) * dΩ_sparse, FE_space.V, FE_space.V₀)

end

function get_parametric_specifics(ROM_info, μ_nb)

  function prepare_model(μ, case)
    if case === 3
      return generate_cartesian_model(ref_info, stretching, μ[1])
    else case === 1
      return DiscreteModelFromFile(ROM_info.paths.mesh_path)
    end
  end
  model = prepare_model(μ_nb, ROM_info.case)

  function prepare_α(x, μ, case)
    if case ===0
      return sum(μ)
    elseif case === 1 || case === 2
      return μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])
    else
      return 1
    end
  end
  α(x) = prepare_α(x, μ_nb, ROM_info.case)

  function prepare_f(x, μ, case)
    if case === 2
      return sin(μ_nb[4] * x[1]) + sin(μ_nb[4] * x[2])
    else
      return 1
    end
  end
  f(x) = prepare_f(x, μ_nb, ROM_info.case)
  h(x) = 1

  ParametricSpecifics(μ_nb, model, α, f, [], h)

end

function ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim, RB_method)
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)
  mesh_path = paths.mesh_path
  FEM_snap_path = paths.FEM_snap_path
  FEM_structures_path = paths.FEM_structures_path
  ROM_path = joinpath(paths.current_test, RB_method)
  create_dir(ROM_path)
  basis_path = joinpath(ROM_path, "basis")
  create_dir(basis_path)
  ROM_structures_path = joinpath(ROM_path, "ROM_structures")
  create_dir(ROM_structures_path)
  gen_coords_path = joinpath(ROM_path, "gen_coords")
  create_dir(gen_coords_path)
  results_path = joinpath(ROM_path, "results")
  create_dir(results_path)
  _ -> (mesh_path, FEM_snap_path, FEM_structures_path, basis_path, ROM_structures_path, gen_coords_path, results_path)
end

function compute_errors(ũ::Array, uₕ::Array, norm_matrix = nothing)
  #=MODIFY
  =#

  mynorm(uₕ - ũ, norm_matrix) / mynorm(uₕ, norm_matrix)

end

function compute_MDEIM_error(problem_info, ROM_info, RB_variables, μ)

  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  Aₙ_μ = (RB_variables.Φₛᵘ)' * assemble_stiffness(FE_space, ROM_info, parametric_info) * RB_variables.Φₛᵘ

end

#= function MPOD(S_sparse::SparseMatrixCSC, ϵ = 1e-5)

  Nₕ = convert(Int64, sqrt(size(S_sparse)[1]))
  z, _, _ = findnz(S_sparse)
  z = unique(z)
  S = Matrix(S_sparse[z, :])
  U = POD(S, ϵ)
  m = size(U)[2]
  _, jU, vU = findnz(sparse(U))

  return sparse(repeat(z, m, 1)[:], jU, vU, Nₕ ^ 2, m)

end

function MDEIM(U_sparse::SparseMatrixCSC)

  m = size(U_sparse)[2]
  z, _, _ = findnz(U_sparse)
  z = unique(z)
  DEIM_mat = Matrix(U_sparse[z, :])
  DEIM_idx = zeros(Int64, m)

  DEIM_idx[1] = convert(Int64, argmax(abs.(DEIM_mat[:, 1]))[1])
  for m in range(2, size(U_sparse)[2])
    res = DEIM_mat[:, m] - DEIM_mat[:, 1:(m-1)] * (DEIM_mat[DEIM_idx[1:(m-1)], 1:(m-1)] \ DEIM_mat[DEIM_idx[1:(m-1)], m])
    DEIM_idx[m] = convert(Int64, argmax(abs.(res))[1])
  end

  return z[DEIM_idx]

end =#
