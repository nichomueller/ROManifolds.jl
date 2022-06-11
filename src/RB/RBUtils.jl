function ROM_paths(root, problem_type, problem_name, mesh_name, RB_method, case, test_case)
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

function build_sparse_mat(FEMInfo::ProblemInfoSteady, FEMSpace::SteadyProblem,
    Param::ParametricInfoSteady, el::Vector; var="A")

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  if var == "A"
    Mat = assemble_matrix(∫(∇(FEMSpace.ϕᵥ) ⋅ (Param.α * ∇(FEMSpace.ϕᵤ))) * dΩ_sparse, FEMSpace.V, FEMSpace.V₀)
  else
    error("Unrecognized sparse matrix")
  end

  Mat

end

function build_sparse_mat(FEMInfo::ProblemInfoUnsteady,
  FEMSpace::UnsteadyProblem,
  Param::ParametricInfoUnsteady,
  el::Vector,
  timesθ::Vector;
  var="A")

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  Nₜ = length(timesθ)

  function define_Matₜ(t::Real, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    elseif mat == "M"
      return assemble_matrix(∫(FEMSpace.ϕᵥ*(Param.m(t)*FEMSpace.ϕᵤ(t)))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    else
      error("Unrecognized sparse matrix")
    end
  end
  Matₜ(t) = define_Matₜ(t, var)

  for (i_t,t) in enumerate(timesθ)
    i,j,v = findnz(Matₜ(t))
    if i_t == 1
      global Mat = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ*Nₜ)
    else
      Mat[:,(i_t-1)*FEMSpace.Nₛᵘ+1:i_t*FEMSpace.Nₛᵘ] =
        sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ)
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

function remove_small_entries(A::Array,tol=1e-15) ::Array
  A[A.<=tol].=0
  A
end

function compute_errors(uₕ::Matrix, RBVars::RBSteadyProblem, norm_matrix = nothing)

  mynorm(uₕ - RBVars.ũ, norm_matrix) / mynorm(uₕ, norm_matrix)

end

function compute_errors(uₕ::Matrix, RBVars::RBUnsteadyProblem, norm_matrix = nothing)

  H1_err = zeros(RBVars.Nₜ)
  H1_sol = zeros(RBVars.Nₜ)

  for i = 1:RBVars.Nₜ
    H1_err[i] = mynorm(uₕ[:, i] - RBVars.S.ũ[:, i], norm_matrix)
    H1_sol[i] = mynorm(uₕ[:, i], norm_matrix)
  end

  return H1_err ./ H1_sol, norm(H1_err) / norm(H1_sol)

end

function compute_MDEIM_error(FEMSpace::FEMProblem, RBInfo::Info, RBVars::RBProblem)

  Aₙ_μ = (RBVars.Φₛᵘ)' * assemble_stiffness(FEMSpace, RBInfo, Param) * RBVars.Φₛᵘ

end
