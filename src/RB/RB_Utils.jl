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

function get_blocks_position(RBVars::RBProblem)
  if typeof(RBVars) ∈ (::PoissonS, ::PoissonST)
    ([1], [1])
  else
    ([1, 2, 3], [1, 2])
  end
end

function assemble_FEM_structure(
  FEMSpace::FEMProblem,
  RBInfo::Info,
  Param::ParamInfo)

  assemble_FEM_structure(FEMSpace, RBInfo.FEMInfo, Param)

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

function ParamInfo(
  RBInfo::Info,
  μ::Vector,
  var::String)

  ParamInfo(RBInfo.FEMInfo, μ, var)

end

function ParamInfo(
  RBInfo::Info,
  μ::Vector)

  ParamInfo(RBInfo.FEMInfo, μ)

end

function ParamFormInfo(
  RBInfo::Info,
  μ::Vector,
  var::String)

  ParamFormInfo(RBInfo.FEMInfo, μ, var)

end

function ParamFormInfo(
  RBInfo::Info,
  μ::Vector)

  ParamFormInfo(RBInfo.FEMInfo, μ)

end

function problem_vectors(RBInfo::Info)
  vecs = ["F", "H", "L", "Lc"]
  intersect(RBInfo.problem_structures, vecs)::Vector{String}
end

function problem_matrices(RBInfo::Info)
  setdiff(RBInfo.problem_structures, problem_vectors(RBInfo))::Vector{String}
end

function get_timesθ(RBInfo::ROMInfoST{T}) where T

  T.(get_timesθ(RBInfo.FEMInfo))

end

function times_dictionary(
  RBInfo::Info,
  offline_time::Float,
  online_time::Float)

  if RBInfo.get_offline_structures
    offline_time = NaN
  end

  Dict("off_time"=>offline_time, "on_time"=>online_time)

end

function initialize_RB_system(RBVars::RBProblem{T}) where T
  RBVars.LHSₙ = Matrix{T}[]
  RBVars.RHSₙ = Matrix{T}[]
  RBVars.xₙ = Matrix{T}[]
end

function initialize_online_time(RBVars::RBProblem)
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
  FEMSpace::FEMProblemS,
  FEMInfo::FEMInfoS,
  Param::ParamInfoS,
  el::Vector{Int})

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(dΩ_sparse, Param)

  assemble_FEM_structure(FEMSpace, FEMInfo, Param)

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

function θ(
  FEMSpace::FEMProblemS{D},
  RBInfo::ROMInfoS{T},
  Param::ParamInfoS,
  MDEIM::MVMDEIM) where {D,T}

  if Param.var ∉ RBInfo.probl_nl
    θ = [[Param.fun(VectorValue(D, T))[1]]]
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
  FEMSpace::FEMProblemS,
  RBVars::RBProblemS,
  MDEIM::MDEIMm,
  var::String) where T

  Fun_μ_sparse =
    assemble_sparse_fun(FEMSpace, FEMInfo, MDEIM.el, var)
  MDEIM_online(RBVars, Fun_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)

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
  xₕ::Matrix{T},
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
