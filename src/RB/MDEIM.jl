include("MV_snapshots.jl")

function MDEIM_POD(S::Matrix{T}, ϵ=1e-5) where T

  U, Σ, Vᵀ = svd(S)
  V = Vᵀ'::Matrix{T}

  energies = cumsum(Σ .^ 2)
  MDEIM_err_bound = vcat(sqrt(norm(inv(U'U))) * Σ[2:end], 0.0) # approx by excess, should be norm(inv(U[idx,:]))

  N₁ = findall(x -> x ≥ (1 - ϵ^2) * energies[end], energies)[1]
  N₂ = findall(x -> x ≤ ϵ, MDEIM_err_bound)[1]
  N = max(N₁, N₂)::Int
  err = max(sqrt(1-energies[N]/energies[end]), MDEIM_err_bound[N])::Float
  println("Basis number obtained via POD is $N, projection error ≤ $err")

  U[:,1:N], V[:,1:N]

end

function MDEIM_offline(Mat::Matrix)

  n = size(Mat)[2]
  idx = Int[]

  append!(idx, Int(argmax(abs.(Mat[:, 1]))))

  @simd for m = 2:n

    res = (Mat[:, m] - Mat[:, 1:m-1] *
      (Mat[idx[1:m-1], 1:m-1] \ Mat[idx[1:m-1], m]))
    append!(idx, convert(Int, argmax(abs.(res))[1]))

  end

  unique!(idx)
  Matᵢ = Mat[idx, :]

  idx, Matᵢ

end

function MDEIM_offline(
  MDEIM::MMDEIM,
  RBInfo::ROMInfoS,
  RBVars::RBS{T},
  var::String) where T

  FEMSpace, μ = get_FEMμ_info(RBInfo)
  Nₛ = get_Nₛ(RBVars, var)

  Mat, row_idx = snaps_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)
  idx_full, Matᵢ = MDEIM_offline(Mat)
  idx = from_full_idx_to_sparse_idx(idx_full, row_idx, Nₛ)
  idx_space, _ = from_vec_to_mat_idx(idx, Nₛ)
  el = find_FE_elements(FEMSpace, idx_space, var)

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.row_idx, MDEIM.el =
    Mat, Matᵢ, idx, row_idx, el

end

function MDEIM_offline(
  MDEIM::VMDEIM,
  RBInfo::ROMInfoS,
  ::RBS{T},
  var::String) where T

  FEMSpace, μ = get_FEMμ_info(RBInfo)

  Mat = snaps_DEIM(FEMSpace, RBInfo, μ, var)
  idx, Matᵢ = MDEIM_offline(Mat)
  el = find_FE_elements(FEMSpace, idx, var)

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.el = Mat, Matᵢ, idx, el

end

function MDEIM_offline(
  MDEIM::MMDEIM,
  RBInfo::ROMInfoST,
  RBVars::RBST{T},
  var::String) where T

  FEMSpace, μ = get_FEMμ_info(RBInfo)
  Nₛ = get_Nₛ(RBVars, var)

  Mat, Mat_time, row_idx = snaps_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)

  idx_full, Matᵢ = MDEIM_offline(Mat)
  idx = from_full_idx_to_sparse_idx(idx_full, row_idx, Nₛ)
  idx_space, _ = from_vec_to_mat_idx(idx, Nₛ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(idx_space))

  time_idx, _ = MDEIM_offline(Mat_time)
  unique!(sort!(time_idx))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.time_idx, MDEIM.row_idx, MDEIM.el =
    Mat, Matᵢ, idx, time_idx, row_idx, el

end

function MDEIM_offline(
  MDEIM::VMDEIM,
  RBInfo::ROMInfoST,
  var::String)

  FEMSpace, μ = get_FEMμ_info(RBInfo)

  Mat, Mat_time = snaps_DEIM(FEMSpace, RBInfo, μ, var)

  idx, Matᵢ = MDEIM_offline(Mat)
  if var == "H"
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Γn, unique(idx))
  elseif var == "Lc"
    el = find_FE_elements(FEMSpace.Q₀, FEMSpace.Ω, unique(idx))
  else
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(idx))
  end

  time_idx, _ = MDEIM_offline(Mat_time)
  unique!(sort!(time_idx))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.time_idx, MDEIM.el =
    Mat, Matᵢ, idx, time_idx, el

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

  if Param.var ∈ RBInfo.affine_structures
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

  if var ∈ RBInfo.affine_structures
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

  if var ∈ RBInfo.affine_structures
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

function MDEIM_online(
  Mat_nonaffine::AbstractArray{T},
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  @fastmath Matᵢ \ reshape(Mat_nonaffine, :, Nₜ)[idx]

end

function MDEIM_online(
  Fun_nonaffine::Function,
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  function θ(u)
    @fastmath Matᵢ \ reshape(Fun_nonaffine(u), :, 1)[idx]
  end

end
