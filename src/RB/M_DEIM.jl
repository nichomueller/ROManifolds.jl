include("MV_snapshots.jl")

function M_DEIM_POD(S::Matrix{T}, ϵ=1e-5) where T

  U, Σ, Vᵀ = svd(S)
  V = Vᵀ'::Matrix{T}

  energies = cumsum(Σ .^ 2)
  M_DEIM_err_bound = vcat(sqrt(norm(inv(U'U))) * Σ[2:end], 0.0) # approx by excess, should be norm(inv(U[MDEIM_idx,:]))

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

function M_DEIM_offline(M_DEIM_mat::Vector{Matrix{T}}) where T

  M_DEIM_idx = Vector{Int}[]
  M_DEIMᵢ_mat = Matrix{T}[]

  for nb = eachindex(M_DEIM_mat)
    M_DEIM_idx_nb, M_DEIMᵢ_mat_nb = M_DEIM_offline(M_DEIM_mat[nb])
    push!(M_DEIM_idx, M_DEIM_idx_nb)
    push!(M_DEIMᵢ_mat, M_DEIMᵢ_mat_nb)
  end

  M_DEIM_idx, M_DEIMᵢ_mat

end

function select_FEM_dim(FEMSpace::FEMProblem, var::String)
  if var ∈ ("B", "Lc")
    FEMSpace.Nₛᵖ
  else
    FEMSpace.Nₛᵘ
  end
end

function MDEIM_offline(
  RBInfo::ROMInfoS,
  ::RBProblemS{T},
  var::String) where T

  println("Building $(RBInfo.nₛ_MDEIM) snapshots of $var")

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Nₕ = select_FEM_dim(FEMSpace, var)

  MDEIM_mat, row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, μ, var)
  MDEIM_idx, MDEIMᵢ_mat = M_DEIM_offline(MDEIM_mat)
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx, row_idx, Nₕ)
  MDEIM_idx_sparse_space, _ = from_vec_to_mat_idx(MDEIM_idx_sparse, Nₕ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(MDEIM_idx_sparse_space))

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, row_idx, el

end

function MDEIM_offline_nonlinear(
  RBInfo::ROMInfoS,
  RBVars::RBProblemS{T},
  var::String) where T

  println("Building $(RBInfo.nₛ_MDEIM * RBVars.nₛᵘ) snapshots of $var")

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Nₕ = select_FEM_dim(FEMSpace, var)

  MDEIM_mat, row_idx = get_snaps_MDEIM_nonlinear(FEMSpace, RBInfo, RBVars, μ, var)
  MDEIM_idx, MDEIMᵢ_mat = M_DEIM_offline(MDEIM_mat)
  MDEIM_mat = blocks_to_matrix(MDEIM_mat)
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx, row_idx, Nₕ)
  MDEIM_idx_sparse_space, _ = from_vec_to_mat_idx(MDEIM_idx_sparse, Nₕ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique_block(MDEIM_idx_sparse_space))

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, row_idx, el

end

function MDEIM_offline(
  RBInfo::ROMInfoST,
  RBVars::RBProblemST{T},
  var::String) where T

  println("Building $(RBInfo.nₛ_MDEIM) snapshots of $var")

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)
  Nₕ = select_FEM_dim(FEMSpace, var)

  MDEIM_mat, MDEIM_mat_time, row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, RBVars, μ, var)

  MDEIM_idx, MDEIMᵢ_mat = M_DEIM_offline(MDEIM_mat)
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx, row_idx, Nₕ)
  MDEIM_idx_sparse_space, _ = from_vec_to_mat_idx(MDEIM_idx_sparse, Nₕ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(MDEIM_idx_sparse_space))

  MDEIM_idx_time, _ = M_DEIM_offline(MDEIM_mat_time)
  unique!(sort!(MDEIM_idx_time))

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, row_idx, el, MDEIM_idx_time

end

function DEIM_offline(RBInfo::ROMInfoS{T}, var::String) where T

  println("Building $(RBInfo.nₛ_DEIM) snapshots of $var")

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)

  DEIM_mat = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)
  DEIM_idx, DEIMᵢ_mat = M_DEIM_offline(DEIM_mat)
  if var == "H"
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Γn, unique(DEIM_idx))
  elseif var == "Lc"
    el = find_FE_elements(FEMSpace.Q₀, FEMSpace.Ω, unique(DEIM_idx))
  else
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(DEIM_idx))
  end

  DEIM_mat, DEIM_idx, DEIMᵢ_mat, el

end

function DEIM_offline(RBInfo::ROMInfoST{T}) where T

  println("Building $(RBInfo.nₛ_DEIM) snapshots of $var")

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)

  DEIM_mat, DEIM_mat_time = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)

  DEIM_idx, DEIMᵢ_mat = M_DEIM_offline(DEIM_mat)
  if var == "H"
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Γn, unique(DEIM_idx))
  elseif var == "Lc"
    el = find_FE_elements(FEMSpace.Q₀, FEMSpace.Ω, unique(DEIM_idx))
  else
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(DEIM_idx))
  end

  DEIM_idx_time, _ = M_DEIM_offline(DEIM_mat_time)
  unique!(sort!(DEIM_idx_time))

  DEIM_mat, DEIM_idx, DEIMᵢ_mat, el, DEIM_idx_time

end

function M_DEIM_online(
  ::RBProblemS,
  Mat_nonaffine::Matrix{T},
  Matᵢ::Matrix{T},
  idx::Vector{Int}) where T

  @fastmath Matᵢ \ Matrix{T}(reshape(Mat_nonaffine, :, 1)[idx, :])

end

function M_DEIM_online(
  ::RBProblemS,
  Mat_nonaffine::SparseMatrixCSC{T, Int},
  Matᵢ::Matrix{T},
  idx::Vector{Int}) where T

  @fastmath Matᵢ \ Matrix{T}(reshape(Mat_nonaffine, :, 1)[idx, :])

end

function M_DEIM_online(
  RBVars::RBProblemST,
  Mat_nonaffine::Matrix{T},
  Matᵢ::Matrix{T},
  idx::Vector{Int}) where T

  @fastmath (Matᵢ \ Matrix{T}(reshape(Mat_nonaffine, :, RBVars.Nₜ)[idx, :]))

end

function M_DEIM_online(
  RBVars::RBProblemST,
  Mat_nonaffine::SparseMatrixCSC{T, Int},
  Matᵢ::Matrix{T},
  idx::Vector{Int}) where T

  @fastmath (Matᵢ \ Matrix{T}(reshape(Mat_nonaffine, :, RBVars.Nₜ)[idx, :]))

end

function M_DEIM_online(
  RBVars::RBProblem,
  Mat_nonaffine,
  Matᵢ::Vector{Matrix{T}},
  idx::Vector{Vector{Int}}) where T

  θ = Matrix{T}[]

  for b = eachindex(Mat_nonaffine)
    push!(θ, M_DEIM_online(RBVars, Mat_nonaffine[b], Matᵢ[b], idx[b]))
  end

  blocks_to_matrix(θ)

end
