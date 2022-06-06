function ROM_paths(root, problem_type, problem_name, mesh_name, RB_method, case; test_case="")
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, case)
  mesh_path = paths.mesh_path
  FEM_snap_path = paths.FEM_snap_path
  FEM_structures_path = paths.FEM_structures_path
  ROM_path = joinpath(paths.current_test, RB_method*test_case)
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

function POD(S, ϵ = 1e-5, X = nothing)

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

  @info "Basis number obtained via POD is $N, projection error ≤ $((sqrt(abs(1 - cumulative_energy / total_energy))))"

  if issparse(U)
    U = Matrix(U)
  end

  if !isnothing(X)
    return Matrix((L' \ U[:, 1:N])[invperm(H.p), :]), Σ
  else
    return U[:, 1:N], Σ
  end

end

function build_sparse_mat(FEMInfo::ProblemInfoSteady, FESpace::SteadyProblem, Param::ParametricInfoSteady, el::Vector; var="A")

  Ω_sparse = view(FESpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  if var == "A"
    Mat = assemble_matrix(∫(∇(FESpace.ϕᵥ) ⋅ (Param.α * ∇(FESpace.ϕᵤ))) * dΩ_sparse, FESpace.V, FESpace.V₀)
  else
    @error "Unrecognized sparse matrix"
  end

  Mat

end

function build_sparse_mat(FEMInfo::ProblemInfoUnsteady,
  FESpace::UnsteadyProblem,
  Param::ParametricInfoUnsteady,
  el::Vector,
  times_θ::Vector;
  var="A")

  Ω_sparse = view(FESpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  Nₜ = length(times_θ)

  function define_Matₜ(t::Real, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FESpace.ϕᵥ) ⋅ (Param.α(t) * ∇(FESpace.ϕᵤ(t)))) * dΩ_sparse, FESpace.V(t), FESpace.V₀)
    elseif mat == "M"
      return assemble_matrix(∫(FESpace.ϕᵥ ⋅ (Param.m(t) * FESpace.ϕᵤ(t))) * dΩ_sparse, FESpace.V(t), FESpace.V₀)
    else
      @error "Unrecognized sparse matrix"
    end
  end
  Matₜ(t) = define_Matₜ(t, var)

  for (i_t,t) in enumerate(times_θ)
    i,j,v = findnz(Matₜ(t))
    if i_t == 1
      global Mat = sparse(i,j,v,FESpace.Nₛᵘ,FESpace.Nₛᵘ*Nₜ)
    else
      Mat[:,(i_t-1)*FESpace.Nₛᵘ+1:i_t*FESpace.Nₛᵘ] = sparse(i,j,v,FESpace.Nₛᵘ,FESpace.Nₛᵘ)
    end
  end

  Mat

end

function blocks_to_matrix(A_block::Array, N_blocks::Int)

  A = zeros(prod(size(A_block[1])), N_blocks)
  for n = 1:N_blocks
    A[:, n] = A_block[n][:]
  end

  A

end

function matrix_to_blocks(A::Array)

  A_block = Matrix{Float64}[]
  N_blocks = size(A)[end]
  dims = Tuple(size(A)[1:end-1])
  order = prod(size(A)[1:end-1])
  for n = 1:N_blocks
    push!(A_block, reshape(A[:][(n-1)*order+1:n*order], dims))
  end

  A_block

end

function compute_errors(uₕ::Vector, RBVars::RBSteadyProblem, norm_matrix = nothing)

  mynorm(uₕ - RBVars.ũ, norm_matrix) / mynorm(uₕ, norm_matrix)

end

function compute_errors(uₕ::Vector, RBVars::RBUnsteadyProblem, norm_matrix = nothing)

  H1_err = zeros(RBVars.Nₜ)
  H1_sol = zeros(RBVars.Nₜ)

  for i = 1:RBVars.Nₜ
    H1_err[i] = mynorm(uₕ[:, i] - RBVars.S.ũ[:, i], norm_matrix)
    H1_sol[i] = mynorm(uₕ[:, i], norm_matrix)
  end

  return H1_err ./ H1_sol, norm(H1_err) / norm(H1_sol)

end

function compute_MDEIM_error(FESpace::FEMProblem, RBInfo::Info, RBVars::RBProblem)

  Aₙ_μ = (RBVars.Φₛᵘ)' * assemble_stiffness(FESpace, RBInfo, Param) * RBVars.Φₛᵘ

end
