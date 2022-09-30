include("MV_snapshots.jl")

function M_DEIM_POD(S::Matrix{T}, ϵ=1e-5) where T

  U, Σ, Vᵀ = svd(S)
  V = Vᵀ'::Matrix{T}

  energies = cumsum(Σ .^ 2)
  M_DEIM_err_bound = vcat(sqrt(norm(inv(U'U))) * Σ[2:end], 0.0) # approx by excess, should be norm(inv(U[idx,:]))

  N₁ = findall(x -> x ≥ (1 - ϵ^2) * energies[end], energies)[1]
  N₂ = findall(x -> x ≤ ϵ, M_DEIM_err_bound)[1]
  N = max(N₁, N₂)::Int
  err = max(sqrt(1-energies[N]/energies[end]),M_DEIM_err_bound[N])::Float
  println("Basis number obtained via POD is $N, projection error ≤ $err")

  U[:,1:N], V[:,1:N]

end

function M_DEIM_offline(M_DEIM_mat::Matrix)

  n = size(M_DEIM_mat)[2]
  M_DEIM_idx = Int[]

  append!(M_DEIM_idx, Int(argmax(abs.(M_DEIM_mat[:, 1]))))

  @simd for m = 2:n

    res = (M_DEIM_mat[:, m] - M_DEIM_mat[:, 1:m-1] *
      (M_DEIM_mat[M_DEIM_idx[1:m-1], 1:m-1] \ M_DEIM_mat[M_DEIM_idx[1:m-1], m]))
    append!(M_DEIM_idx, convert(Int, argmax(abs.(res))[1]))

    if abs(det(M_DEIM_mat[M_DEIM_idx[1:m], 1:m])) ≤ 1e-80
      error("Something went wrong with the construction of (M)DEIM basis:
        obtaining singular nested matrices")
    end

  end

  unique!(M_DEIM_idx)
  M_DEIMᵢ_mat = M_DEIM_mat[M_DEIM_idx, :]

  M_DEIM_idx, M_DEIMᵢ_mat

end

function select_FEM_dim(FEMSpace::FEMProblem, var::String)
  if var ∈ ("B", "Lc")
    FEMSpace.Nₛᵖ
  else
    FEMSpace.Nₛᵘ
  end
end

function MDEIM_offline!(
  MDEIM::MDEIMmS,
  RBInfo::ROMInfoS,
  RBVars::RBProblemS{T},
  var::String) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Nₕ = select_FEM_dim(FEMSpace, var)

  Mat, row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)
  idx_full, Matᵢ = M_DEIM_offline(Mat)
  idx = from_full_idx_to_sparse_idx(idx_full, row_idx, Nₕ)
  idx_space, _ = from_vec_to_mat_idx(idx, Nₕ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(idx_space))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.row_idx, MDEIM.el =
    Mat, Matᵢ, idx, row_idx, el

end

function MDEIM_offline!(
  MDEIM::MDEIMmST,
  RBInfo::ROMInfoST,
  RBVars::RBProblemST{T},
  var::String) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Nₕ = select_FEM_dim(FEMSpace, var)

  Mat, Mat_time, row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)

  idx_full, Matᵢ = M_DEIM_offline(Mat)
  idx = from_full_idx_to_sparse_idx(idx_full, row_idx, Nₕ)
  idx_space, _ = from_vec_to_mat_idx(idx, Nₕ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(idx_space))

  idx_time, _ = M_DEIM_offline(Mat_time)
  unique!(sort!(idx_time))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.idx_time, MDEIM.row_idx, MDEIM.el =
    Mat, Matᵢ, idx, idx_time, row_idx, el

end

function MDEIM_offline!(
  MDEIM::MDEIMvS,
  RBInfo::ROMInfoS{T},
  var::String) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)

  Mat = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)
  idx, Matᵢ = M_DEIM_offline(Mat)
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
  MDEIM::MDEIMvST,
  RBInfo::ROMInfoST{T},
  var::String) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)

  Mat, Mat_time = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)

  idx, Matᵢ = M_DEIM_offline(Mat)
  if var == "H"
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Γn, unique(idx))
  elseif var == "Lc"
    el = find_FE_elements(FEMSpace.Q₀, FEMSpace.Ω, unique(idx))
  else
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(idx))
  end

  idx_time, _ = M_DEIM_offline(Mat_time)
  unique!(sort!(idx_time))

  MDEIM.Mat, MDEIM.Matᵢ, MDEIM.idx, MDEIM.idx_time, MDEIM.el =
    Mat, Matᵢ, idx, idx_time, el

end

function M_DEIM_online(
  ::RBProblemS,
  Mat_nonaffine::AbstractArray{T},
  Matᵢ::Matrix{T},
  idx::Vector{Int}) where T

  @fastmath Matᵢ \ Matrix{T}(reshape(Mat_nonaffine, :, 1)[idx, :])

end

function M_DEIM_online(
  RBVars::RBProblemST,
  Mat_nonaffine::AbstractArray{T},
  Matᵢ::Matrix{T},
  idx::Vector{Int}) where T

  @fastmath Matᵢ \ Matrix{T}(reshape(Mat_nonaffine, :, RBVars.Nₜ)[idx, :])

end

function M_DEIM_online(
  ::RBProblemS,
  Fun_nonaffine::Function,
  Matᵢ::Matrix{T},
  idx::Vector{Int}) where T

  function θ(u)
    @fastmath Matᵢ \ Matrix{T}(reshape(Fun_nonaffine(u), :, 1)[idx, :])
  end

end
