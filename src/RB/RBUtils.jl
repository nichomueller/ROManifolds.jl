function ROM_paths(root, problem_steadiness, problem_name, mesh_name, RB_method, case)
  paths = FEM_paths(root, problem_steadiness, problem_name, mesh_name, case)
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

function select_RB_method(
  RB_method::String,
  tol::String,
  add_info::Dict) ::String

  if add_info["nested_POD"]
    RB_method *= "_nest"
  end
  if add_info["st_M_DEIM"]
    RB_method *= "_st"
  end
  if add_info["fun_M_DEIM"]
    RB_method *= "_fun"
  end
  if add_info["sampl_M_DEIM"]
    RB_method *= "_sampl"
  end

  RB_method *= tol

end

function get_method_id(problem_name::String, RB_method::String)
  if problem_name == "poisson" && RB_method == "S-GRB"
    return (0,)
  elseif problem_name == "poisson" && RB_method == "S-PGRB"
    return (0,0)
  elseif problem_name == "poisson" && RB_method == "ST-GRB"
    return (0,0,0)
  elseif problem_name == "poisson" && RB_method == "ST-PGRB"
    return (0,0,0,0)
  elseif problem_name == "stokes" && RB_method == "ST-GRB"
    return (0,0,0,0,0)
  elseif problem_name == "stokes" && RB_method == "ST-PGRB"
    return (0,0,0,0,0,0)
  else
    error("unimplemented")
  end
end

function assemble_FEM_structure(
  FEMSpace::FEMProblem,
  RBInfo::ROMInfoSteady,
  Param::ParametricInfo,
  var::String)

  assemble_FEM_structure(FEMSpace,RBInfo.FEMInfo,Param,var)

end

function assemble_FEM_structure(
  FEMSpace::FEMProblem,
  RBInfo::ROMInfoUnsteady,
  Param::ParametricInfo,
  var::String)

  assemble_FEM_structure(FEMSpace,RBInfo.FEMInfo,Param,var)

end

function get_ParamInfo(RBInfo::Info, μ::Vector)

  get_ParamInfo(RBInfo.FEMInfo, RBInfo.FEMInfo.problem_id, μ)

end

function get_timesθ(RBInfo::ROMInfoUnsteady)

  get_timesθ(RBInfo.FEMInfo)

end

function build_sparse_mat(
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady,
  el::Vector{Int64};
  var="A")

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)

  if var == "A"
    Mat = assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α*∇(FEMSpace.ϕᵤ)))*dΩ_sparse,
      FEMSpace.V, FEMSpace.V₀)
  else
    error("Unrecognized sparse matrix")
  end

  Mat::SparseMatrixCSC{T, Int64}

end

function build_sparse_mat(
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo{T},
  Param::ParametricInfoUnsteady,
  el::Vector{Int64},
  timesθ::Vector;
  var="A") where T

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  Nₜ = length(timesθ)

  function define_Matₜ(t::Real, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    elseif var == "M"
      return assemble_matrix(∫(FEMSpace.ϕᵥ*(Param.m(t)*FEMSpace.ϕᵤ(t)))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    else
      error("Unrecognized sparse matrix")
    end
  end
  Matₜ(t) = define_Matₜ(t, var)

  for (i_t,t) in enumerate(timesθ)
    i,j,v = findnz(Matₜ(t))::Tuple{Vector{Int},Vector{Int},Vector{T}}
    if i_t == 1
      global Mat = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ*Nₜ)
    else
      Mat[:,(i_t-1)*FEMSpace.Nₛᵘ+1:i_t*FEMSpace.Nₛᵘ] =
        sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ)
    end
  end

  Mat::SparseMatrixCSC{T, Int64}

end

function blocks_to_matrix(A_block::Array{T}, N_blocks::Int64) where T

  A = zeros(T,prod(size(A_block[1])), N_blocks)
  for n = 1:N_blocks
    A[:, n] = A_block[n][:]
  end

  A

end

function matrix_to_blocks(A::Array{T}) where T

  A_block = Matrix{T}[]
  N_blocks = size(A)[end]
  dims = Tuple(size(A)[1:end-1])
  order = prod(size(A)[1:end-1])
  for n = 1:N_blocks
    push!(A_block, reshape(A[:][(n-1)*order+1:n*order], dims))
  end

  A_block

end

function remove_small_entries(A::Array,tol=1e-15)
  A[A.<=tol].=0
  A
end

function compute_errors(
  uₕ::Vector{T},
  RBVars::RBSteadyProblem{T},
  norm_matrix = nothing) where T

  mynorm(uₕ - RBVars.ũ, norm_matrix) / mynorm(uₕ, norm_matrix)

end

function compute_errors(
  uₕ::Matrix,
  RBVars::RBUnsteadyProblem{T},
  norm_matrix = nothing) where T

  H1_err = zeros(T, RBVars.Nₜ)
  H1_sol = zeros(T, RBVars.Nₜ)

  @simd for i = 1:RBVars.Nₜ
    H1_err[i] = mynorm(uₕ[:, i] - RBVars.S.ũ[:, i], norm_matrix)
    H1_sol[i] = mynorm(uₕ[:, i], norm_matrix)
  end

  return H1_err ./ H1_sol, norm(H1_err) / norm(H1_sol)

end
