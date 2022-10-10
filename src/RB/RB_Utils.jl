function ROM_paths(FEMPaths, RB_method)

  ROM_path = joinpath(FEMPaths.current_test, RB_method)
  create_dir(ROM_path)
  ROM_structures_path = joinpath(ROM_path, "ROM_structures")
  create_dir(ROM_structures_path)
  results_path = joinpath(ROM_path, "results")
  create_dir(results_path)

  ROMPath(FEMPaths, ROM_structures_path, results_path)

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

function get_affine_entries(
  operators::Vector{String},
  affine_names::NTuple{D}) where D

  affine_entries = Int[]
  for idx = 1:D
    if (affine_names[idx]) ∈ operators .* "ₙ"
      append!(affine_entries, idx)
    end
  end

  affine_entries

end

function assemble_FEM_structure(
  FEMSpace::FEMProblem,
  RBInfo::ROMInfoS,
  Param::ParamInfoS,
  var::String)

  assemble_FEM_structure(FEMSpace,RBInfo.FEMInfo,Param,var)

end

function assemble_FEM_structure(
  FEMSpace::FEMProblem,
  RBInfo::ROMInfoST,
  Param::ParamInfoST,
  var::String)

  assemble_FEM_structure(FEMSpace,RBInfo.FEMInfo,Param,var)

end

function get_Φₛ(RBVars::RBProblem, var::String)
  if var ∈ ("B", "Lc")
    RBVars.Φₛ[2]
  else
    RBVars.Φₛ[1]
  end
end

function get_Nₛ(RBVars::RBProblem, var::String)
  if var ∈ ("B", "Lc")
    RBVars.Nₛ[2]
  else
    RBVars.Nₛ[1]
  end
end

function get_ParamInfo(
  RBInfo::Info,
  μ::Vector,
  var::String)

  get_ParamInfo(RBInfo.FEMInfo, μ, var)

end

function get_ParamInfo(
  RBInfo::Info,
  μ::Vector)

  get_ParamInfo(RBInfo.FEMInfo, μ)

end

function get_ParamFormInfo(
  RBInfo::Info,
  μ::Vector,
  var::String)

  get_ParamFormInfo(RBInfo.FEMInfo, μ, var)

end

function get_ParamFormInfo(
  RBInfo::Info,
  μ::Vector)

  get_ParamFormInfo(RBInfo.FEMInfo, μ)

end

function get_timesθ(RBInfo::ROMInfoST{T}) where T

  T.(get_timesθ(RBInfo.FEMInfo))

end

function initialize_RB_system(RBVars::RBProblemS{T}) where T
  RBVars.LHSₙ = Matrix{T}[]
  RBVars.RHSₙ = Matrix{T}[]
end

function initialize_RB_system(RBVars::RBProblemST{T}) where T
  RBVars.LHSₙ = Matrix{T}[]
  RBVars.RHSₙ = Matrix{T}[]
end

function initialize_online_time(RBVars::RBProblem)
  RBVars.online_time = 0.0
end

function assemble_ith_row_MatΦ(
  Mat::Matrix{T},
  Φₛ::Matrix{T},
  r_idx::Vector{Int},
  c_idx::Vector{Int},
  i::Int) where T

  sparse_idx = findall(x -> x == i, r_idx)
  Matrix(reshape((Mat[sparse_idx,:]' * Φₛ[c_idx[sparse_idx],:])', 1, :))
end

function assemble_sparse_mat(
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS,
  el::Vector{Int},
  var::String)

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)

  function define_Mat(FEMSpace::FEMSpacePoissonS, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α*∇(FEMSpace.ϕᵤ)))*dΩ_sparse,
        FEMSpace.V, FEMSpace.V₀)
    else
      error("Unrecognized sparse matrix")
    end
  end
  function define_Mat(FEMSpace::FEMSpaceStokesS, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α*∇(FEMSpace.ϕᵤ)))*dΩ_sparse,
        FEMSpace.V, FEMSpace.V₀)
    elseif var == "B"
      return assemble_matrix(∫(FEMSpace.ψᵧ*(Param.b*∇⋅(FEMSpace.ϕᵤ)))*dΩ_sparse,
        FEMSpace.V, FEMSpace.Q₀)
    else
      error("Unrecognized sparse matrix")
    end
  end
  function define_Mat(FEMSpace::FEMSpaceNavierStokesS, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α*∇(FEMSpace.ϕᵤ)))*dΩ_sparse,
        FEMSpace.V, FEMSpace.V₀)
    elseif var == "B"
      return assemble_matrix(∫(FEMSpace.ψᵧ*(Param.b*∇⋅(FEMSpace.ϕᵤ)))*dΩ_sparse,
        FEMSpace.V, FEMSpace.Q₀)
    else
      error("Unrecognized sparse matrix")
    end
  end

  define_Mat(FEMSpace, var)::SparseMatrixCSC{Float, Int}

end

function assemble_sparse_mat(
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST,
  el::Vector{Int},
  timesθ::Vector,
  var::String)

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  Nₛ = get_Nₛ(RBVars, var)
  Nₜ = length(timesθ)

  function define_Matₜ(FEMSpace::FEMSpacePoissonST, t::Real, var::String)
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
  function define_Matₜ(FEMSpace::FEMSpaceStokesST, t::Real, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    elseif var == "B"
      return assemble_matrix(∫(FEMSpace.ψᵧ*(Param.b(t)*∇⋅(FEMSpace.ϕᵤ)))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.Q₀)
    elseif var == "M"
      return assemble_matrix(∫(FEMSpace.ϕᵥ⋅(Param.m(t)*FEMSpace.ϕᵤ(t)))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    else
      error("Unrecognized sparse matrix")
    end
  end
  function define_Matₜ(FEMSpace::FEMSpaceNavierStokesST, t::Real, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⊙(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    elseif var == "B"
      return assemble_matrix(∫(FEMSpace.ψᵧ*(Param.b(t)*∇⋅(FEMSpace.ϕᵤ)))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.Q₀)
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
      Mat = sparse(i,j,v,Nₛ,FEMSpace.Nₛᵘ*Nₜ)
    else
      Mat[:,(i_t-1)*FEMSpace.Nₛᵘ+1:i_t*FEMSpace.Nₛᵘ] =
        sparse(i,j,v,Nₛ,FEMSpace.Nₛᵘ)
    end
  end

  Mat::SparseMatrixCSC{Float, Int}

end

function assemble_sparse_vec(
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS,
  el::Vector{Int},
  var::String)

  if var == "H"
    Ω_sparse = view(FEMSpace.Γn, el)
  else
    Ω_sparse = view(FEMSpace.Ω, el)
  end
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)

  function define_Vec(FEMSpace::FEMSpacePoissonS, var::String)
    if var == "F"
      return assemble_vector(∫(FEMSpace.ϕᵥ*Param.f)*dΩ_sparse, FEMSpace.V₀)
    elseif var == "H"
      return assemble_vector(∫(FEMSpace.ϕᵥ*Param.h)*dΩ_sparse, FEMSpace.V₀)
    elseif var == "L"
      g = define_g_FEM(FEMSpace, Param)
      return assemble_vector(
        ∫(Param.α * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g))*dΩ_sparse,FEMSpace.V₀)
    else
      error("Unrecognized variable")
    end
  end
  function define_Vec(FEMSpace::FEMSpaceStokesS, var::String)
    if var == "F"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f)*dΩ_sparse, FEMSpace.V₀)
    elseif var == "H"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h)*dΩ_sparse, FEMSpace.V₀)
    elseif var == "L"
      g = define_g_FEM(FEMSpace, Param)
      return assemble_vector(
        ∫(Param.α * ∇(FEMSpace.ϕᵥ) ⊙ ∇(g))*dΩ_sparse,FEMSpace.V₀)
    elseif var == "Lc"
      g = define_g_FEM(FEMSpace, Param)
      return assemble_vector(
        ∫(FEMSpace.ψᵧ * (∇⋅g))*dΩ_sparse,FEMSpace.Q₀)
    else
      error("Unrecognized variable")
    end
  end
  function define_Vec(FEMSpace::FEMSpaceNavierStokesS, var::String)
    if var == "F"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f)*dΩ_sparse, FEMSpace.V₀)
    elseif var == "H"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h)*dΩ_sparse, FEMSpace.V₀)
    elseif var == "L"
      g = define_g_FEM(FEMSpace, Param)
      return assemble_vector(
        ∫(Param.α * ∇(FEMSpace.ϕᵥ) ⊙ ∇(g))*dΩ_sparse,FEMSpace.V₀)
    elseif var == "Lc"
      g = define_g_FEM(FEMSpace, Param)
      return assemble_vector(
        ∫(FEMSpace.ψᵧ * (∇⋅g))*dΩ_sparse,FEMSpace.Q₀)
    else
      error("Unrecognized variable")
    end
  end

  define_Vec(FEMSpace, var)::Vector{Float}

end

function assemble_sparse_vec(
  FEMSpace::FEMProblemST,
  FEMInfo::FEMInfoST,
  Param::ParamInfoST,
  el::Vector{Int},
  timesθ::Vector,
  var::String)

  if var == "H"
    Ω_sparse = view(FEMSpace.Γn, el)
  else
    Ω_sparse = view(FEMSpace.Ω, el)
  end
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)

  function define_Vecₜ(FEMSpace::FEMSpacePoissonST, t::Real, var::String)
    if var == "F"
      return assemble_vector(∫(FEMSpace.ϕᵥ*Param.f(t))*dΩ_sparse, FEMSpace.V₀)
    elseif var == "H"
      return assemble_vector(∫(FEMSpace.ϕᵥ*Param.h(t))*dΩ_sparse, FEMSpace.V₀)
    elseif var == "L"
      g = define_g_FEM(FEMSpace, Param)
      return assemble_vector(
        ∫(Param.α(t) * ∇(FEMSpace.ϕᵥ) ⋅ ∇(g(t)))*dΩ_sparse,FEMSpace.V₀)
    else
      error("Unrecognized variable")
    end
  end

  function define_Vecₜ(FEMSpace::FEMSpaceStokesST, t::Real, var::String)
    if var == "F"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.f(t))*dΩ_sparse, FEMSpace.V₀)
    elseif var == "H"
      return assemble_vector(∫(FEMSpace.ϕᵥ⋅Param.h(t))*dΩ_sparse, FEMSpace.V₀)
    elseif var == "L"
      g = define_g_FEM(FEMSpace, Param)
      return assemble_vector(
        ∫(Param.α(t) * ∇(FEMSpace.ϕᵥ) ⊙ ∇(g(t)))*dΩ_sparse,FEMSpace.V₀)
    elseif var == "Lc"
      g = define_g_FEM(FEMSpace, Param)
      return assemble_vector(
        ∫(FEMSpace.ψᵧ * (∇⋅g(t)))*dΩ_sparse,FEMSpace.Q₀)
    else
      error("Unrecognized variable")
    end
  end

  Vecₜ(t) = define_Vecₜ(FEMSpace, t, var)

  Vec = zeros(FEMSpace.Nₛᵘ, length(timesθ))
  for (i_t,t) in enumerate(timesθ)
    Vec[:, i_t] = Vecₜ(t)
  end

  Vec::Matrix{Float}

end

function assemble_sparse_fun(
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  el::Vector{Int},
  var::String)

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)

  function define_Mat(u)
    if var == "C"
      (assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
        (∇(FEMSpace.ϕᵤ)'⋅u) )*dΩ_sparse, FEMSpace.V, FEMSpace.V₀))
    elseif var == "D"
      (assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
        (∇(u)'⋅FEMSpace.ϕᵤ) )*dΩ_sparse, FEMSpace.V, FEMSpace.V₀))
    else
      error("Unrecognized sparse matrix")
    end
  end

  define_Mat::Function

end

function interpolated_θ(
  RBVars::RBProblemST{T},
  Mat_μ_sparse::AbstractArray,
  timesθ::Vector{T},
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  time_idx::Vector{Int}) where T

  red_timesθ = timesθ[time_idx]
  discarded_time_idx = setdiff(collect(1:RBVars.Nₜ), time_idx)
  θ = zeros(T, length(idx), RBVars.Nₜ)

  red_θ = (Matᵢ \
    Matrix{T}(reshape(Mat_μ_sparse, :, length(red_timesθ))[idx, :]))

  etp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ,1,:), red_θ')
  θ[:, time_idx] = red_θ
  for iₜ = discarded_time_idx
    θ[:, iₜ] = ScatteredInterpolation.evaluate(etp,[timesθ[iₜ]])
  end

  θ::Matrix{T}

end

function get_scalar_value(
  val,
  T::Type)

  if typeof(val) != T
    T.(val[1][1])
  else
    T.(val)
  end

end

function θ!(
  θ::Vector{Vector{T}},
  FEMSpace::FEMProblemS{D},
  RBInfo::ROMInfoS{T},
  RBVars::RBProblemS,
  Param::ParamInfoS,
  fun::Function,
  MDEIM::MDEIMm,
  var::String) where {D,T}

  if var ∉ RBInfo.probl_nl
    push!(θ, [get_scalar_value(fun(VectorValue(D, T)), T)])
  else
    Mat_μ_sparse =
      assemble_sparse_mat(FEMSpace, FEMInfo, Param, MDEIM.el, var)::SparseMatrixCSC{Float, Int}
    θvec = M_DEIM_online(RBVars, Mat_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
    θ = [[θvec[q]] for q in eachindex(θvec)]
  end

  θ

end

function θ!(
  θ::Vector{Vector{T}},
  FEMSpace::FEMProblemS{D},
  RBInfo::ROMInfoS{T},
  RBVars::RBProblemS,
  Param::ParamInfoS,
  fun::Function,
  MDEIM::MDEIMv,
  var::String) where {D,T}

  if var ∉ RBInfo.probl_nl
    push!(θ, [get_scalar_value(fun(VectorValue(D, T)), T)])
  else
    Vec_μ_sparse =
      assemble_sparse_vec(FEMSpace, FEMInfo, Param, MDEIM.el, var)::Vector{Float}
    θvec = M_DEIM_online(RBVars, Vec_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
    θ = [[θvec[q]] for q in eachindex(θvec)]
  end

  θ

end

function θ!(
  θ::Vector{Vector{T}},
  FEMSpace::FEMProblemST{D},
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  Param::ParamInfoST,
  fun::Function,
  MDEIM::MDEIMm,
  var::String) where {D,T}

  timesθ = get_timesθ(RBInfo)

  if var ∉ RBInfo.probl_nl
    for t_θ in timesθ
      push!(θ, [get_scalar_value(fun(VectorValue(D, T), t_θ), T)])
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[MDEIM.time_idx]
      Mat_μ_sparse = assemble_sparse_mat(
        FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ, var)
      θmat = interpolated_θ(RBVars, Mat_μ_sparse, timesθ, MDEIM.Matᵢ,
        MDEIM.idx, MDEIM.time_idx)
    else
      Mat_μ_sparse = assemble_sparse_mat(
        FEMSpace, FEMInfo, Param, MDEIM.el, timesθ, var)
      θmat = M_DEIM_online(RBVars, Mat_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
    end
    θ = [[θmat[q, :]] for q in size(θmat)[1]]
  end

  θ

end

function θ!(
  θ::Vector{Vector{T}},
  FEMSpace::FEMProblemST{D},
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  Param::ParamInfoST,
  fun::Function,
  MDEIM::MDEIMv,
  var::String) where {D,T}

  timesθ = get_timesθ(RBInfo)

  if var ∉ RBInfo.probl_nl
    for t_θ in timesθ
      push!(θ, [get_scalar_value(fun(VectorValue(D, T), t_θ), T)])
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[MDEIM.time_idx]
      Vec_μ_sparse = T.(assemble_sparse_vec(
        FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ, var))
      θmat = interpolated_θ(RBVars, Vec_μ_sparse, timesθ, MDEIM.Matᵢ,
        MDEIM.idx, MDEIM.time_idx)
    else
      Vec_μ_sparse = assemble_sparse_vec(FEMSpace, FEMInfo, Param, MDEIM.el, timesθ, var)
      θmat = M_DEIM_online(RBVars, Vec_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
    end
    θ = [[θmat[q, :]] for q in size(θmat)[1]]
  end

  θ

end

function θ_function(
  FEMSpace::FEMProblemS,
  RBVars::RBProblemS,
  MDEIM::MDEIMm,
  var::String) where T

  Fun_μ_sparse =
    assemble_sparse_fun(FEMSpace, FEMInfo, MDEIM.el, var)
  M_DEIM_online(RBVars, Fun_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)

end

function compute_errors(
  xₕ::Vector{T},
  x̃::Vector{T},
  X::Matrix{T}) where T

  mynorm(xₕ - x̃, X) / mynorm(xₕ, X)

end

function compute_errors(
  xₕ::Matrix{T},
  x̃::Matrix{T},
  X::Matrix{T}) where T

  @assert size(xₕ)[2] == size(x̃)[2] == 1 "Something is wrong"
  compute_errors(xₕ[:, 1], x̃[:, 1], X)

end

function compute_errors(
  xₕ::Vector{T},
  x̃::Matrix{T},
  X::Matrix{T},
  Nₜ::Int) where T

  norm_err = zeros(T, Nₜ)
  norm_sol = zeros(T, Nₜ)

  @simd for i = 1:Nₜ
    norm_err[i] = mynorm(xₕ[:, i] - x̃[:, i], X)
    norm_sol[i] = mynorm(xₕ[:, i], X)
  end

  norm_err ./ norm_sol, norm(norm_err) / norm(norm_sol)

end
