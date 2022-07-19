function ROM_paths(FEMPaths, RB_method)

  ROM_path = joinpath(FEMPaths.current_test, RB_method)
  create_dir(ROM_path)
  ROM_structures_path = joinpath(ROM_path, "ROM_structures")
  create_dir(ROM_structures_path)
  results_path = joinpath(ROM_path, "results")
  create_dir(results_path)

  RBPathInfo(FEMPaths, ROM_structures_path, results_path)

end

function get_mesh_path(RBInfo::Info)
  RBInfo.Paths.FEMPaths.mesh_path
end

function get_FEM_snap_path(RBInfo::Info)
  RBInfo.Paths.FEMPaths.FEM_snap_path
end

function get_FEM_structures_path(RBInfo::Info)
  RBInfo.Paths.FEMPaths.FEM_structures_path
end

function select_RB_method(
  RB_method::String,
  tol::String,
  add_info::Dict) ::String

  if add_info["st_M_DEIM"]
    RB_method *= "_st"
  end
  if add_info["fun_M_DEIM"]
    RB_method *= "_fun"
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
  Param::ParametricInfoSteady,
  var::String)

  assemble_FEM_structure(FEMSpace,RBInfo.FEMInfo,Param,var)

end

function assemble_FEM_structure(
  FEMSpace::FEMProblem,
  RBInfo::ROMInfoUnsteady,
  Param::ParametricInfoUnsteady,
  var::String)

  assemble_FEM_structure(FEMSpace,RBInfo.FEMInfo,Param,var)

end

function get_ParamInfo(RBInfo::Info, μ::Vector{T}) where T

  get_ParamInfo(RBInfo.FEMInfo, μ)

end

function get_timesθ(RBInfo::ROMInfoUnsteady{T}) where T

  T.(get_timesθ(RBInfo.FEMInfo))

end

function assemble_parametric_structure(
  θ::Matrix{T},
  Mat::Array{T,D}) where {T,D}

  Mat_shape = size(Mat)
  Mat = reshape(Mat,:,Mat_shape[end])

  if size(θ)[2] > 1
    Matμ = reshape(Mat*θ,(Mat_shape[1:end-1]...,size(θ)[2]))
  else
    Matμ = reshape(Mat*θ,Mat_shape[1:end-1])
  end

  Matμ::Array{T,D-1}

end

function build_sparse_mat(
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady,
  el::Vector{Int};
  var="A")

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)

  function define_Mat(::FEMSpacePoissonSteady, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α*∇(FEMSpace.ϕᵤ)))*dΩ_sparse,
        FEMSpace.V, FEMSpace.V₀)
    else
      error("Unrecognized sparse matrix")
    end
  end
  function define_Mat(::FEMSpaceStokesSteady, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α*∇(FEMSpace.ϕᵤ)))*dΩ_sparse,
        FEMSpace.V, FEMSpace.V₀)
    else
      error("Unrecognized sparse matrix")
    end
  end

  define_Mat(FEMSpace, var)::SparseMatrixCSC{Float, Int}

end

function build_sparse_mat(
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady,
  el::Vector{Int},
  timesθ::Vector;
  var="A")

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  Nₜ = length(timesθ)

  function define_Matₜ(::FEMSpacePoissonUnsteady, t::Real, var::String)
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
  function define_Matₜ(::FEMSpaceStokesUnsteady, t::Real, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    elseif var == "M"
      return assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Param.m(t)*FEMSpace.ϕᵤ(t)))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    else
      error("Unrecognized sparse matrix")
    end
  end
  Matₜ(t) = define_Matₜ(FEMSpace, t, var)

  Mat = sparse([], [], Float[])
  for (i_t,t) in enumerate(timesθ)
    i,j,v = findnz(Matₜ(t))::Tuple{Vector{Int},Vector{Int},Vector{Float}}
    if i_t == 1
      Mat = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ*Nₜ)
    else
      Mat[:,(i_t-1)*FEMSpace.Nₛᵘ+1:i_t*FEMSpace.Nₛᵘ] =
        sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ)
    end
  end

  Mat::SparseMatrixCSC{Float, Int}

end

function build_sparse_vec(
  FEMSpace::SteadyProblem,
  FEMInfo::SteadyInfo,
  Param::ParametricInfoSteady,
  el::Vector{Int};
  var="F")

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)

  function define_Vec(::FEMSpacePoissonSteady, var::String)
    if var == "F"
      return assemble_vector(∫(FEMSpace.ϕᵥ*Param.f)*dΩ_sparse, FEMSpace.V₀)
    elseif var == "H"
      return assemble_vector(∫(FEMSpace.ϕᵥ*Param.h)*dΩ_sparse, FEMSpace.V₀)
    end
  end
  function define_Vec(::FEMSpaceStokesSteady, var::String)
    if var == "F"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f)*dΩ_sparse, FEMSpace.V₀)
    elseif var == "H"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h)*dΩ_sparse, FEMSpace.V₀)
    end
  end

  define_Vec(FEMSpace, var)::Vector{Float}

end

function build_sparse_vec(
  FEMSpace::UnsteadyProblem,
  FEMInfo::UnsteadyInfo,
  Param::ParametricInfoUnsteady,
  el::Vector{Int},
  timesθ::Vector;
  var="F")

  if var == "F"
    Ω_sparse = view(FEMSpace.Ω, el)
    dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  elseif var == "H"
    Ω_sparse = view(FEMSpace.Γn, el)
    dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  else
    error("Unrecognized variable")
  end

  function define_Vecₜ(::FEMSpacePoissonUnsteady, t::Real, var::String)
    if var == "F"
      return assemble_vector(∫(FEMSpace.ϕᵥ*Param.f(t))*dΩ_sparse, FEMSpace.V₀)
    else var == "H"
      return assemble_vector(∫(FEMSpace.ϕᵥ*Param.h(t))*dΩ_sparse, FEMSpace.V₀)
    end
  end
  function define_Vecₜ(::FEMSpaceStokesUnsteady, t::Real, var::String)
    if var == "F"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f(t))*dΩ_sparse, FEMSpace.V₀)
    else var == "H"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h(t))*dΩ_sparse, FEMSpace.V₀)
    end
  end
  Vecₜ(t) = define_Vecₜ(FEMSpace, t, var)

  Vec = zeros(FEMSpace.Nₛᵘ, length(timesθ))
  for (i_t,t) in enumerate(timesθ)
    Vec[:, i_t] = Vecₜ(t)
  end

  Vec::Matrix{Float}

end

function blocks_to_matrix(A_block::Array{T}, N_blocks::Int) where T

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
  RBVars::RBUnsteadyProblem{T},
  uₕ::Matrix,
  ũ::Matrix,
  norm_matrix = nothing) where T

  norm_err = zeros(T, RBVars.Nₜ)
  norm_sol = zeros(T, RBVars.Nₜ)

  @simd for i = 1:RBVars.Nₜ
    norm_err[i] = mynorm(uₕ[:, i] - ũ[:, i], norm_matrix)
    norm_sol[i] = mynorm(uₕ[:, i], norm_matrix)
  end

  return norm_err ./ norm_sol, norm(norm_err) / norm(norm_sol)

end
