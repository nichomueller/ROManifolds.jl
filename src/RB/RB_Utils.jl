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

function get_ParamInfo(RBInfo::Info, μ::Vector{T}) where T

  get_ParamInfo(RBInfo.FEMInfo, μ)

end

function get_ParamInfo(RBInfo::Info, FEMSpace::FEMProblem, μ::Vector{T}) where T

  get_ParamInfo(RBInfo.FEMInfo, FEMSpace, μ)

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
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  RBVars::RBProblemS,
  ::ParamInfoS,
  el::Vector{Vector{Int}},
  var::String)

  function define_Mat(
    FEMSpace::FEMSpaceNavierStokesS,
    Φₖ::Vector,
    dΩ_sparse::Measure,
    var::String)

    if var == "C"
      Φₖ_fun = FEFunction(FEMSpace.V₀, Φₖ)
      return assemble_matrix(∫( FEMSpace.ϕᵥ ⊙
        (∇(FEMSpace.ϕᵤ)' ⋅ Φₖ_fun) )*dΩ_sparse, FEMSpace.V, FEMSpace.V₀)
    else
      error("Unrecognized sparse matrix")
    end
  end

  Mat = SparseMatrixCSC[]

  @assert length(el) == RBVars.nₛᵘ

  for b = eachindex(el)
    Ω_sparse = view(FEMSpace.Ω, el[b])
    dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
    push!(Mat, define_Mat(FEMSpace, RBVars.Φₛᵘ[:,b], dΩ_sparse, var)::SparseMatrixCSC{Float, Int})
  end

  Mat

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

#= function assemble_RBapprox_convection(
  FEMSpace::FEMSpaceNavierStokesS,
  Param::ParamInfoS,
  RBVars::NavierStokesS{T}) where T

  C = assemble_convection(FEMSpace, Param)

  Cᵩ = Array{T}(undef,0,0,0)
  for k = 1:RBVars.nₛᵘ_quad
    Φₛᵘ_fun = FEFunction(FEMSpace.V₀_quad, RBVars.Φₛᵘ_quad[:, k])
    i, v = findnz(C(Φₛᵘ_fun)[:])::Tuple{Vector{Int},Vector{T}}
    if k == 1
      RBVars.row_idx_C = i
      Cᵩ = zeros(T, length(RBVars.row_idx_C), 1, RBVars.nₛᵘ)
    end
    Cᵩ[:, :, k] = v
  end

  Cᵩ

end

function assemble_RBapprox_convection(
  FEMSpace::FEMSpaceNavierStokesST,
  Param::ParamInfoST,
  RBInfo::ParamNavierStokesST,
  RBVars::NavierStokesST{T}) where T

  function index_mapping_inverse_quad(i::Int)
    iₛ = 1+Int(floor((i-1)/RBVars.nₜᵘ_quad))
    iₜ = i-(iₛ-1)*RBVars.nₜᵘ_quad
    iₛ, iₜ
  end

  C = assemble_convection(FEMSpace, Param)
  timesθ = get_timesθ(RBInfo)

  Cᵩ = Array{T}(undef,0,0,0)
  for nₜ = 1:RBVars.Nₜ
    for k = 1:RBVars.nᵘ
      kₛ, kₜ = index_mapping_inverse(k)
      Φₛᵘ_fun = FEFunction(FEMSpace.V₀_quad,
        RBVars.Φₛᵘ_quad[:, kₛ] * RBVars.Φₜᵘ_quad[nₜ, kₜ])
      # this is wrong: Φₛᵘ_fun is not at time timesθ[nₜ]
      i, v = findnz(C(Φₛᵘ_fun, timesθ[nₜ])[:])::Tuple{Vector{Int},Vector{T}}
      if k*nₜ == 1
        RBVars.row_idx_C = i
        Cᵩ = zeros(T, length(RBVars.row_idx_C), RBVars.Nₜ, RBVars.nᵘ)
      end
      Cᵩ[:, nₜ, k] = v
    end
  end

  Cᵩ

end =#

function interpolated_θ(
  RBVars::RBProblemST{T},
  Mat_μ_sparse::SparseMatrixCSC{T, Int},
  timesθ::Vector{T},
  MDEIMᵢ::Matrix{T},
  MDEIM_idx::Vector{Int},
  MDEIM_idx_time::Vector{Int}) where T

  red_timesθ = timesθ[MDEIM_idx_time]
  discarded_idx_time = setdiff(collect(1:RBVars.Nₜ), MDEIM_idx_time)
  θ = zeros(T, length(MDEIM_idx), RBVars.Nₜ)

  red_θ = (MDEIMᵢ \
    Matrix{T}(reshape(Mat_μ_sparse, :, length(red_timesθ))[MDEIM_idx, :]))

  etp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ,1,:), red_θ')
  θ[:, MDEIM_idx_time] = red_θ
  for iₜ = discarded_idx_time
    θ[:, iₜ] = ScatteredInterpolation.evaluate(etp,[timesθ[iₜ]])
  end

  θ::Matrix{T}

end

function interpolated_θ(
  RBVars::RBProblemST{T},
  Vec_μ_sparse::Matrix{T},
  timesθ::Vector{T},
  DEIMᵢ::Matrix{T},
  DEIM_idx::Vector{Int},
  DEIM_idx_time::Vector{Int}) where T

  red_timesθ = timesθ[DEIM_idx_time]
  discarded_idx_time = setdiff(collect(1:RBVars.Nₜ), DEIM_idx_time)
  θ = zeros(T, length(MDEIM_idx), RBVars.Nₜ)

  red_θ = (DEIMᵢ \
    Matrix{T}(reshape(Vec_μ_sparse, :, length(red_timesθ))[DEIM_idx, :]))

  etp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ,1,:), red_θ')
  θ[:, DEIM_idx_time] = red_θ
  for iₜ = discarded_idx_time
    θ[:, iₜ] = ScatteredInterpolation.evaluate(etp,[timesθ[iₜ]])
  end

  θ::Matrix{T}

end

function modify_fun(
  fun::Function,
  D::Int,
  T::Type)

  val = fun(zero(VectorValue(D, T)))
  if typeof(val) != T
    val = val[1][1]
  end

  T.(val)

end

function modify_fun(
  fun::Function,
  t_θ::Vector,
  T::Type)

  val = fun(t_θ)
  if typeof(val) != T
    val = val[1][1]
  end

  T.(val)

end

function modify_fun(
  fun::Function,
  μ::Vector,
  t_θ::Vector,
  T::Type)

  val = fun(t_θ, μ)
  if typeof(val) != T
    val = val[1][1]
  end

  T.(val)

end

function θ_matrix(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::RBProblemS,
  Param::ParamInfoS,
  fun::Function,
  MDEIMᵢ::AbstractArray,
  MDEIM_idx::AbstractArray,
  sparse_el::AbstractArray,
  var::String) where T

  if var ∉ RBInfo.probl_nl
    θ = reshape([modify_fun(fun, FEMInfo.D, T)], 1, 1)
  else
    if var == "C"
      Mat_μ_sparse =
        assemble_sparse_mat(FEMSpace, FEMInfo, RBVars, Param, sparse_el, var)
    else
      Mat_μ_sparse =
        assemble_sparse_mat(FEMSpace, FEMInfo, Param, sparse_el, var)
    end
    θ = M_DEIM_online(RBVars, Mat_μ_sparse, MDEIMᵢ, MDEIM_idx)
  end

  θ::Matrix{T}

end

function θ_matrix(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  Param::ParamInfoST,
  fun::Function,
  MDEIMᵢ::Matrix,
  MDEIM_idx::Vector{Int},
  sparse_el::Vector{Int},
  MDEIM_idx_time::Vector{Int},
  var::String) where T

  timesθ = get_timesθ(RBInfo)

  if var ∉ RBInfo.probl_nl
    θ = zeros(T, 1, RBVars.Nₜ)
    for (i_t, t_θ) = enumerate(timesθ)
      θ[i_t] = modify_fun(fun, Param.μ, t_θ, T)
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[MDEIM_idx_time]
      Mat_μ_sparse = assemble_sparse_mat(
        FEMSpace, FEMInfo, Param, sparse_el, red_timesθ, var)
      θ = interpolated_θ(RBVars, Mat_μ_sparse, timesθ, MDEIMᵢ,
        MDEIM_idx, MDEIM_idx_time)
    else
      Mat_μ_sparse = assemble_sparse_mat(
        FEMSpace, FEMInfo, Param, sparse_el,timesθ, var)
      θ = M_DEIM_online(RBVars, Mat_μ_sparse, MDEIMᵢ, MDEIM_idx)
    end
  end

  θ::Matrix{T}

end

function θ_vector(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  ::RBProblemS,
  Param::ParamInfoS,
  fun::Function,
  DEIMᵢ::Matrix,
  DEIM_idx::Vector{Int},
  sparse_el::Vector{Int},
  var::String) where T

  if var ∉ RBInfo.probl_nl
    θ = reshape([modify_fun(fun, FEMInfo.D, T)], 1, 1)
  else
    Vec_μ_sparse =
      T.(assemble_sparse_vec(FEMSpace, FEMInfo, Param, sparse_el, var))
    θ = M_DEIM_online(RBVars, Vec_μ_sparse, DEIMᵢ, DEIM_idx)
  end

  θ::Matrix{T}

end

function θ_vector(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST{T},
  RBVars::RBProblemST,
  Param::ParamInfoST,
  fun::Function,
  DEIMᵢ::Matrix,
  DEIM_idx::Vector{Int},
  sparse_el::Vector{Int},
  DEIM_idx_time::Vector{Int},
  var::String) where T

  timesθ = get_timesθ(RBInfo)

  if var ∉ RBInfo.probl_nl
    θ = zeros(T, 1, RBVars.Nₜ)
    for (i_t, t_θ) = enumerate(timesθ)
      θ[i_t] = modify_fun(fun, t_θ, T) # VERY UGLY - CHANGE NEEDED
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[DEIM_idx_time]
      Vec_μ_sparse = T.(assemble_sparse_vec(
        FEMSpace,FEMInfo, Param, sparse_el, red_timesθ, var))
      θ = interpolated_θ(RBVars, Vec_μ_sparse, timesθ, DEIMᵢ,
        DEIM_idx, DEIM_idx_time)
    else
      Vec_μ_sparse = assemble_sparse_vec(FEMSpace, FEMInfo, Param, sparse_el, timesθ, var)
      θ = M_DEIM_online(RBVars, Vec_μ_sparse, DEIMᵢ, DEIM_idx)
    end
  end

  θ::Matrix{T}

end

function compute_errors(
  ::RBProblemS{T},
  uₕ::Vector,
  ũ::Matrix{T},
  norm_matrix = nothing) where T

  mynorm(uₕ - ũ[:, 1], norm_matrix) / mynorm(uₕ, norm_matrix)

end

function compute_errors(
  RBVars::RBProblemST{T},
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
