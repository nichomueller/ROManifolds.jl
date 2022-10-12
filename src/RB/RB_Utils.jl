function get_mesh_path(RBInfo::ROMInfo)
  RBInfo.Paths.FEMPaths.mesh_path
end

function get_FEM_snap_path(RBInfo::ROMInfo)
  RBInfo.Paths.FEMPaths.FEM_snap_path
end

function get_FEM_structures_path(RBInfo::ROMInfo)
  RBInfo.Paths.FEMPaths.FEM_structures_path
end

function select_RB_method(
  RB_method::String,
  tol::String,
  add_info::Dict) ::String

  if add_info["st_MDEIM"]
    RB_method *= "_st"
  end
  if add_info["fun_MDEIM"]
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

function get_blocks_position(RBVars::RB)
  if typeof(RBVars) ∈ (::PoissonS, ::PoissonST)
    ([1], [1])
  else
    ([1, 2, 3], [1, 2])
  end
end

function assemble_FEM_structure(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  fun::Function,
  var::String)

  Param = ParamInfo(FEMSpace, fun, var)
  assemble_FEM_structure(FEMSpace, RBInfo.FEMInfo, Param)

end

function assemble_FEM_structure(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  Param::ParamInfo)

  assemble_FEM_structure(FEMSpace, RBInfo.FEMInfo, Param)

end

function assemble_FEM_structure(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  μ::Vector{T},
  var::String) where T

  Param = ParamInfo(RBInfo, μ, var)
  assemble_FEM_structure(FEMSpace, RBInfo, Param)

end

function assemble_FEM_structure(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  μvec::Vector{Vector{T}},
  var::String) where T

  MV(μ) = assemble_FEM_structure(FEMSpace, RBInfo, μ, var)
  Broadcasting(MV)(μvec)

end

function get_Φₛ(RBVars::RB, var::String)
  if var ∈ ("B", "Lc")
    Φₛ_left = RBVars.Φₛ[2]
  else
    Φₛ_left = RBVars.Φₛ[1]
  end
  Φₛ_right = RBVars.Φₛ[1]
  Φₛ_left, Φₛ_right
end

function get_Nₛ(RBVars::RB, var::String)
  if var ∈ ("B", "Lc")
    RBVars.Nₛ[2]
  else
    RBVars.Nₛ[1]
  end
end

function ParamInfo(
  RBInfo::ROMInfo,
  μ::Vector,
  var::String)

  ParamInfo(RBInfo.FEMInfo, μ, var)

end

function ParamInfo(
  RBInfo::ROMInfo,
  μ::Vector)

  ParamInfo(RBInfo.FEMInfo, μ)

end

function ParamFormInfo(
  RBInfo::ROMInfo,
  μ::Vector,
  var::String)

  ParamFormInfo(RBInfo.FEMInfo, μ, var)

end

function ParamFormInfo(
  RBInfo::ROMInfo,
  μ::Vector)

  ParamFormInfo(RBInfo.FEMInfo, μ)

end

function get_FEM_vectors(RBInfo::ROMInfo)
  get_FEM_vectors(RBInfo.FEMInfo)
end

function get_FEM_matrices(RBInfo::ROMInfo)
  get_FEM_matrices(RBInfo.FEMInfo)
end

function get_timesθ(RBInfo)
  get_timesθ(RBInfo.FEMInfo)
end

function times_dictionary(
  RBInfo::ROMInfo,
  offline_time::Float,
  online_time::Float)

  if RBInfo.get_offline_structures
    offline_time = NaN
  end

  Dict("off_time"=>offline_time, "on_time"=>online_time)

end

function initialize_RB_system(RBVars::RB{T}) where T
  RBVars.LHSₙ = Matrix{T}[]
  RBVars.RHSₙ = Matrix{T}[]
  RBVars.xₙ = Matrix{T}[]
end

function initialize_online_time(RBVars::RB)
  RBVars.online_time = 0.0
end

function assemble_termsₙ(
  Vars::Vector{MVVariable},
  Params::Vector{ParamInfo},
  operators::Vector{String})

  mult = Broadcasting(.*)

  function assemble_termₙ(var::String)
    Var = MVVariable(Vars, var)
    Param = ParamInfo(Params, var)
    if var ∈ ("L", "Lc")
      -sum(mult(Var.Matₙ, Param.θ))
    else
      sum(mult(Var.Matₙ, Param.θ))
    end
  end

  Broadcasting(assemble_termₙ)(operators)

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

function assemble_sparse_structure(
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS,
  el::Vector{Int})

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(dΩ_sparse, Param)

  assemble_FEM_structure(FEMSpace, FEMInfo, Param)

end

function assemble_sparse_fun(
  FEMSpace::FOMS,
  FEMInfo::FOMInfoS,
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
  RBVars::RBST{T},
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

function θ(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS,
  Param::ParamInfoS,
  MDEIM::MVMDEIM) where D

  if Param.var ∉ RBInfo.affine_structures
    θ = [[Param.fun(VectorValue(D, Float))[1]]]
  else
    Mat_μ_sparse =
      assemble_sparse_structure(FEMSpace, FEMInfo, Param, MDEIM.el)
    θvec = MDEIM_online(Mat_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
    θ = [[θvec[q]] for q in eachindex(θvec)]
  end

  θ::Vector{Vector{T}}

end

function θ!(
  θ::Vector{Vector{T}},
  FEMSpace::FOMST{D},
  RBInfo::ROMInfoST,
  RBVars::RBST,
  Param::ParamInfoST,
  fun::Function,
  MDEIM::MMDEIM,
  var::String) where {D,T}

  timesθ = get_timesθ(RBInfo)

  if var ∉ RBInfo.affine_structures
    for t_θ in timesθ
      push!(θ, [get_scalar_value(fun(VectorValue(D, T), t_θ), T)])
    end
  else
    if RBInfo.st_MDEIM
      red_timesθ = timesθ[MDEIM.time_idx]
      Mat_μ_sparse = assemble_sparse_mat(
        FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ, var)
      θmat = interpolated_θ(RBVars, Mat_μ_sparse, timesθ, MDEIM.Matᵢ,
        MDEIM.idx, MDEIM.time_idx)
    else
      Mat_μ_sparse = assemble_sparse_mat(
        FEMSpace, FEMInfo, Param, MDEIM.el, timesθ, var)
      θmat = MDEIM_online(RBVars, Mat_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
    end
    θ = [[θmat[q, :]] for q in size(θmat)[1]]
  end

  θ

end

function θ!(
  θ::Vector{Vector{T}},
  FEMSpace::FOMST{D},
  RBInfo::ROMInfoST,
  RBVars::RBST,
  Param::ParamInfoST,
  fun::Function,
  MDEIM::VMDEIM,
  var::String) where {D,T}

  timesθ = get_timesθ(RBInfo)

  if var ∉ RBInfo.affine_structures
    for t_θ in timesθ
      push!(θ, [get_scalar_value(fun(VectorValue(D, T), t_θ), T)])
    end
  else
    if RBInfo.st_MDEIM
      red_timesθ = timesθ[MDEIM.time_idx]
      Vec_μ_sparse = T.(assemble_sparse_vec(
        FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ, var))
      θmat = interpolated_θ(RBVars, Vec_μ_sparse, timesθ, MDEIM.Matᵢ,
        MDEIM.idx, MDEIM.time_idx)
    else
      Vec_μ_sparse = assemble_sparse_vec(FEMSpace, FEMInfo, Param, MDEIM.el, timesθ, var)
      θmat = MDEIM_online(RBVars, Vec_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
    end
    θ = [[θmat[q, :]] for q in size(θmat)[1]]
  end

  θ

end

function θ_function(
  FEMSpace::FOMS,
  RBVars::RBS,
  MDEIM::MMDEIM,
  var::String) where T

  Fun_μ_sparse =
    assemble_sparse_fun(FEMSpace, FEMInfo, MDEIM.el, var)
  MDEIM_online(RBVars, Fun_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)

end

function compute_errors(
  xₕ::Vector{T},
  x̃::Vector{T},
  X::Matrix{T}) where T

  norm(xₕ - x̃, X) / norm(xₕ, X)

end

function compute_errors(
  xₕ::Matrix{T},
  x̃::Matrix{T},
  X::Matrix{T}) where T

  @assert size(xₕ)[2] == size(x̃)[2] == 1 "Something is wrong"
  compute_errors(xₕ[:, 1], x̃[:, 1], X)

end

function compute_errors(
  xₕ::Matrix{T},
  x̃::Matrix{T},
  X::Matrix{T},
  Nₜ::Int) where T

  norm_err = zeros(T, Nₜ)
  norm_sol = zeros(T, Nₜ)

  @simd for i = 1:Nₜ
    norm_err[i] = norm(xₕ[:, i] - x̃[:, i], X)
    norm_sol[i] = norm(xₕ[:, i], X)
  end

  norm_err ./ norm_sol, norm(norm_err) / norm(norm_sol)

end
