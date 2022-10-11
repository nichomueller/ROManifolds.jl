include("MV_snapshots.jl")

function MDEIM_POD(S::Matrix{T}, ϵ=1e-5) where T

  U, Σ, Vᵀ = svd(S)
  V = Vᵀ'::Matrix{T}

  energies = cumsum(Σ .^ 2)
  MDEIM_err_bound = vcat(sqrt(norm(inv(U'U))) * Σ[2:end], 0.0) # approx by excess, should be norm(inv(U[idx,:]))

  N₁ = findall(x -> x ≥ (1 - ϵ^2) * energies[end], energies)[1]
  N₂ = findall(x -> x ≤ ϵ, MDEIM_err_bound)[1]
  N = max(N₁, N₂)::Int
  err = max(sqrt(1-energies[N]/energies[end]),MDEIM_err_bound[N])::Float
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

function MDEIM_offline!(
  MDEIM::MDEIMm,
  RBInfo::ROMInfoS,
  RBVars::RBProblemS{T},
  var::String) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Nₛ = get_Nₛ(RBVars, var)

  Mat, row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)
  idx_full, Matᵢ = MDEIM_offline(Mat)
  idx = from_full_idx_to_sparse_idx(idx_full, row_idx, Nₛ)
  idx_space, _ = from_vec_to_mat_idx(idx, Nₛ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(idx_space))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.row_idx, MDEIM.el =
    Mat, Matᵢ, idx, row_idx, el

end

function MDEIM_offline!(
  MDEIM::MDEIMm,
  RBInfo::ROMInfoST,
  RBVars::RBProblemST{T},
  var::String) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Nₛ = get_Nₛ(RBVars, var)

  Mat, Mat_time, row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)

  idx_full, Matᵢ = MDEIM_offline(Mat)
  idx = from_full_idx_to_sparse_idx(idx_full, row_idx, Nₛ)
  idx_space, _ = from_vec_to_mat_idx(idx, Nₛ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(idx_space))

  time_idx, _ = MDEIM_offline(Mat_time)
  unique!(sort!(time_idx))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.time_idx, MDEIM.row_idx, MDEIM.el =
    Mat, Matᵢ, idx, time_idx, row_idx, el

end

function MDEIM_offline!(
  MDEIM::MDEIMv,
  RBInfo::ROMInfoS{T},
  var::String) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)

  Mat = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)
  idx, Matᵢ = MDEIM_offline(Mat)
  if var == "H"
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Γn, unique(idx))
  elseif var == "Lc"
    el = find_FE_elements(FEMSpace.Q₀, FEMSpace.Ω, unique(idx))
  else
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(idx))
  end

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.el = Mat, Matᵢ, idx, el

end

function MDEIM_offline!(
  MDEIM::MDEIMv,
  RBInfo::ROMInfoST{T},
  var::String) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)

  Mat, Mat_time = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)

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
