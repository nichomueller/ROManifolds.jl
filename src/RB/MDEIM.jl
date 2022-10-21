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

  @inbounds for m = 2:n
    res = (Mat[:, m] - Mat[:, 1:m-1] *
      (Mat[idx[1:m-1], 1:m-1] \ Mat[idx[1:m-1], m]))
    append!(idx, Int(argmax(abs.(res))[1]))
  end

  unique!(idx)
  Matᵢ = Mat[idx, :]

  idx, Matᵢ

end

function MDEIM_offline(
  MDEIM::MMDEIM{T},
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  var::String) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))
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
  MDEIM::VMDEIM{T},
  RBInfo::ROMInfoS{ID},
  ::ROMS{ID,T},
  var::String) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  Mat = snaps_MDEIM(FEMSpace, RBInfo, μ, var)
  idx, Matᵢ = MDEIM_offline(Mat)
  el = find_FE_elements(FEMSpace, idx, var)

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.el = Mat, Matᵢ, idx, el

end

function MDEIM_offline(
  MDEIM::MMDEIM{T},
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  var::String) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))
  Nₛ = get_Nₛ(RBVars, var)

  Mat, Mat_time, row_idx = snaps_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)
  idx_full, Matᵢ = MDEIM_offline(Mat)
  idx = from_full_idx_to_sparse_idx(idx_full, row_idx, Nₛ)
  idx_space, _ = from_vec_to_mat_idx(idx, Nₛ)
  el = find_FE_elements(FEMSpace, idx_space, var)

  time_idx, _ = MDEIM_offline(Mat_time)
  unique!(sort!(time_idx))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.time_idx, MDEIM.row_idx, MDEIM.el =
    Mat, Matᵢ, idx, time_idx, row_idx, el

end

function MDEIM_offline(
  MDEIM::VMDEIM{T},
  RBInfo::ROMInfoST{ID},
  ::ROMST{ID,T},
  var::String) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))

  Mat, Mat_time = snaps_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)
  idx, Matᵢ = MDEIM_offline(Mat)
  el = find_FE_elements(FEMSpace, idx_space, var)

  time_idx, _ = MDEIM_offline(Mat_time)
  unique!(sort!(time_idx))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.time_idx, MDEIM.el =
    Mat, Matᵢ, idx, time_idx, el

end

function assemble_sparse_matrix(
  FEMSpace::FOMS{D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfoS,
  el::Vector{Int}) where {ID,D}

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_sparse)

  assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm)

end

function assemble_sparse_vector(
  FEMSpace::FOMS{D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfoS,
  el::Vector{Int}) where {ID,D}

  triang = Gridap.FESpaces.get_triangulation(FEMSpace, Param.var)
  Ω_sparse = view(triang, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_sparse)

  assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm)

end

function assemble_sparse_function(
  FEMSpace::FOMS{D},
  FEMInfo::FOMInfoS{ID},
  Param::ParamInfoS,
  el::Vector{Int}) where {ID,D}

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_sparse)

  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, ParamForm)

end

function assemble_sparse_matrix(
  FEMSpace::FOMST{D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfoST,
  el::Vector{Int},
  timesθ::Vector{T}) where {ID,D,T}

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_sparse)

  assemble_FEM_matrix(FEMSpace, FEMInfo, ParamForm)(timesθ)

end

function assemble_sparse_vector(
  FEMSpace::FOMST{D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfoST,
  el::Vector{Int},
  timesθ::Vector{T}) where {ID,D,T}

  triang = Gridap.FESpaces.get_triangulation(FEMSpace, Param.var)
  Ω_sparse = view(triang, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_sparse)

  assemble_FEM_vector(FEMSpace, FEMInfo, ParamForm)(timesθ)

end

function assemble_sparse_function(
  FEMSpace::FOMST{D},
  FEMInfo::FOMInfoST{ID},
  Param::ParamInfoST,
  el::Vector{Int},
  timesθ::Vector{T}) where {ID,D,T}

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  ParamForm = ParamFormInfo(Param, dΩ_sparse)

  assemble_FEM_nonlinear_matrix(FEMSpace, FEMInfo, ParamForm)(timesθ)

end

function θ(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{ID},
  Param::ParamInfoS,
  MDEIM::MMDEIM{T}) where {ID,D,T}

  if Param.var ∈ RBInfo.affine_structures
    θ = [[Param.fun(VectorValue(D, T))[1]]]
  else
    Mat_μ_sparse =
      assemble_sparse_matrix(FEMSpace, FEMInfo, Param, MDEIM.el)
    θ = MDEIM_online(Mat_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
  end

  θ::Vector{Vector{T}}

end

function θ(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{ID},
  Param::ParamInfoS,
  MDEIM::VMDEIM{T}) where {ID,D,T}

  if Param.var ∈ RBInfo.affine_structures
    θ = [[Param.fun(VectorValue(D, T))[1]]]
  else
    Vec_μ_sparse =
      assemble_sparse_vector(FEMSpace, FEMInfo, Param, MDEIM.el)
    θ = MDEIM_online(Vec_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
  end

  θ::Vector{Vector{T}}

end

function θ_function(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{ID},
  Param::ParamInfoS,
  MDEIM::MMDEIM{T}) where {ID,D,T}

  @assert isnonlinear(RBInfo, Param.var) "This method is only for nonlinear variables"

  Fun_μ_sparse =
    assemble_sparse_function(FEMSpace, FEMInfo, Param, MDEIM.el)
  MDEIM_online(Fun_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)

end

function interpolate_θ(
  Mat_μ_sparse::AbstractArray,
  MDEIM::MVMDEIM{T},
  timesθ::Vector{T}) where T

  idx, time_idx = MDEIM.idx, MDEIM.time_idx
  red_timesθ = timesθ[time_idx]
  discarded_time_idx = setdiff(eachindex(timesθ), time_idx)
  θ = zeros(T, length(idx), length(timesθ))

  red_θ = (MDEIM.Matᵢ \
    Matrix{T}(reshape(Mat_μ_sparse, :, length(red_timesθ))[idx, :]))
  etp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ, 1, :), red_θ')
  θ[:, time_idx] = red_θ
  for iₜ = discarded_time_idx
    θ[:, iₜ] = ScatteredInterpolation.evaluate(etp,[timesθ[iₜ]])
  end

  [[θ[q, :]] for q in size(θ)[1]]

end

function θ(
  FEMSpace::FOMST{D},
  RBInfo::ROMInfoST{ID},
  Param::ParamInfoST,
  MDEIM::MMDEIM{T}) where {ID,D,T}

  timesθ = get_timesθ(RBInfo)

  if Param.var ∈ RBInfo.affine_structures
    θ = [[Param.fun(VectorValue(D, T), tθ)[1]] for tθ in timesθ]
  else
    if RBInfo.st_MDEIM
      red_timesθ = timesθ[MDEIM.time_idx]
      Mat_μ_sparse = assemble_sparse_matrix(
        FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ)
      θ = interpolate_θ(Mat_μ_sparse, MDEIM, timesθ)
    else
      Mat_μ_sparse = assemble_sparse_matrix(
        FEMSpace, FEMInfo, Param, MDEIM.el, timesθ)
      θ = MDEIM_online(RBVars, Mat_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
    end
  end

  θ::Vector{Vector{T}}

end

function θ(
  FEMSpace::FOMST{D},
  RBInfo::ROMInfoST{ID},
  Param::ParamInfoST,
  MDEIM::MMDEIM{T}) where {ID,D,T}

  timesθ = get_timesθ(RBInfo)

  if Param.var ∈ RBInfo.affine_structures
    θ = [[Param.fun(VectorValue(D, T), tθ)[1]] for tθ in timesθ]
  else
    if RBInfo.st_MDEIM
      red_timesθ = timesθ[MDEIM.time_idx]
      Vec_μ_sparse = assemble_sparse_vector(
        FEMSpace, FEMInfo, Param, MDEIM.el, red_timesθ)
      θ = interpolate_θ(Vec_μ_sparse, MDEIM, timesθ)
    else
      Vec_μ_sparse = assemble_sparse_vector(
        FEMSpace, FEMInfo, Param, MDEIM.el, timesθ)
      θ = MDEIM_online(RBVars, Vec_μ_sparse, MDEIM.Matᵢ, MDEIM.idx)
    end
  end

  θ::Vector{Vector{T}}

end

function MDEIM_online(
  Mat_nonaffine::AbstractArray{T},
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  θvec = Matᵢ \ reshape(Mat_nonaffine, :, Nₜ)[idx,:]
  matrix_to_vecblocks(Matrix{T}(θvec'))

end

function MDEIM_online(
  Fun_nonaffine::Function,
  Matᵢ::Matrix{T},
  idx::Vector{Int},
  Nₜ=1) where T

  θmat(u) = Matᵢ \ reshape(Fun_nonaffine(u), :, Nₜ)[idx,:]
  θblock(u) = matrix_to_vecblocks(Matrix{T}(θmat(u)'))

  θblock

end
