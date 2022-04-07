include("../utils/general.jl")
include("../FEM/FEM_utils.jl")

function POD(S, ϵ = 1e-5, X = nothing)
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

  while cumulative_energy / total_energy < 1.0 - ϵ ^ 2 && N < size(S̃)[2]
    N += 1
    cumulative_energy += Σ[N] ^ 2
    @info "POD loop number $N, cumulative energy = $cumulative_energy"
  end

  @info "Basis number obtained via POD is $N, projection error ≤ $(sqrt(1 - cumulative_energy / total_energy))"

  if issparse(U)
    U = Matrix(U)
  end

  if !isnothing(X)
    return Matrix((L' \ U[:, 1:N])[invperm(H.p), :]) #Matrix((L' \ U[:, 1:N]))
  else
    return U[:, 1:N]
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

  @info "Basis number obtained via POD is $N, projection error <= $sqrt(1-(cumulative_energy / total_energy))"
  V = Q * U[:, N]

  if !isnothing(X)
    return Matrix((L' \ V[:, 1:N])[invperm(H.p), :])
  else
    return V
  end

end


function DEIM_offline(S, ϵ = 1e-5, save_path = nothing)
  #=MODIFY
  =#

  DEIM_mat = POD(S, ϵ)

  (N, n) = size(DEIM_mat)
  DEIM_idx = zeros(Int64, n)

  DEIM_idx[1] = convert(Int64, argmax(abs.(DEIM_mat[:, 1]))[1])
  for m in range(2, n)
    res = DEIM_mat[:, m] - DEIM_mat[:, 1:(m-1)] * (DEIM_mat[DEIM_idx[1:(m-1)], 1:(m-1)] \ DEIM_mat[DEIM_idx[1:(m-1)], m])
    DEIM_idx[m] = convert(Int64, argmax(abs.(res))[1])
  end

  if !isnothing(save_path)
    @info "Offline phase of DEIM and MDEIM are the same: be careful with the path to which the (M)DEIM matrix and indices are saved"
    save_CSV(DEIM_mat, save_path)
    save_CSV(DEIM_idx, save_path)
  end

  (DEIM_mat, DEIM_idx)

end


function DEIM_online(vec_nonaffine, DEIM_mat, DEIM_idx)
  #=MODIFY
  =#

  vec_affine = zeros(size(vec_nonaffine))

  DEIM_coeffs = DEIM_mat[DEIM_idx[:], :] \ vec_nonaffine[DEIM_idx[:]]
  mul!(vec_affine, DEIM_mat, DEIM_coeffs)

  return DEIM_coeffs, vec_affine

end


function MDEIM_online(mat_nonaffine, MDEIM_mat, MDEIM_idx)
  #=MODIFY
  S is already in the correct format, so it is a matrix of size (R*C, quantity), while mat_nonaffine is of size (R, C)
  =#

  (R, C) = size(mat_nonaffine)
  vec_affine = zeros(R * C, 1)

  MDEIM_coeffs = MDEIM_mat[MDEIM_idx[:], :] \ reshape(mat_nonaffine, R * C, 1)[MDEIM_idx[:]]
  mul!(vec_affine, MDEIM_mat, MDEIM_coeffs)
  mat_affine = reshape(vec_affine, R, C)

  return MDEIM_coeffs, mat_affine

end

function get_parametric_specifics(ROM_info, μ_nb)

  function prepare_α(x, μ, case)
    if case === 1
      return μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])
    elseif case === 2
      return μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])
    else
      return sum(μ)
    end
  end
  α(x) = prepare_α(x, μ_nb, ROM_info.case)

  if ROM_info.case === 0
    return parametric_specifics(μ_nb, [], α, [], [], [])
  elseif ROM_info.case === 1
    return parametric_specifics(μ_nb, [], α, [], [], [])
  elseif ROM_info.case === 2
    f(x) = sin(μ_nb[4] * x[1]) + sin(μ_nb[4] * x[2])
    g(x) = sin(μ_nb[5] * x[1]) + sin(μ_nb[5] * x[2])
    return parametric_specifics(μ_nb, [], α, f, g, [])
  else
    model = generate_cartesian_model(ref_info, stretching, μ)
    return parametric_specifics(μ_nb, model, [], [], [], [])
  end

end

function ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim, RB_method)
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)
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
  _ -> (FEM_snap_path, FEM_structures_path, basis_path, ROM_structures_path, gen_coords_path, results_path)
end

function compute_errors(RB_variables::RB_problem, uₕ_test::Array)
  #=MODIFY
  =#

  mynorm(uₕ_test - RB_variables.ũ, RB_variables.Xᵘ) / mynorm(uₕ_test, RB_variables.Xᵘ)

end
