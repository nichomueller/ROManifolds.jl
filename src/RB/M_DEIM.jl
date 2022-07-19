include("MV_snapshots.jl")

function M_DEIM_POD(S::Matrix{T}, ϵ=1e-5) where T

  M_DEIM_mat, Σ, _ = svd(S)

  energies = cumsum(Σ .^ 2)
  mult_factor = sqrt(size(S)[2]) * norm(inv(M_DEIM_mat'M_DEIM_mat))
  M_DEIM_err_bound = vcat(mult_factor * Σ[2:end], 0.0)

  energies = cumsum(Σ .^ 2)
  N₁ = findall(x -> x ≥ (1 - ϵ^2) * energies[end], energies)[1]
  N₂ = findall(x -> x ≤ ϵ, M_DEIM_err_bound)[1]
  N = max(N₁, N₂)::Int
  err = max(sqrt(1-energies[N]/energies[end]),M_DEIM_err_bound[N])::Float
  println("Basis number obtained via POD is $N, projection error ≤ $err")

  T.(M_DEIM_mat[:, 1:N]), T.(Σ)

end


function M_DEIM_offline(M_DEIM_mat::Matrix, Σ::Vector)

  (N, n) = size(M_DEIM_mat)
  M_DEIM_idx = Int[]
  append!(M_DEIM_idx, Int(argmax(abs.(M_DEIM_mat[:, 1]))))
  @simd for m = 2:n
    res = (M_DEIM_mat[:, m] -
           M_DEIM_mat[:, 1:m-1] * (M_DEIM_mat[M_DEIM_idx[1:m-1], 1:m-1] \
                                   M_DEIM_mat[M_DEIM_idx[1:m-1], m]))
    append!(M_DEIM_idx, convert(Int, argmax(abs.(res))[1]))
    if abs(det(M_DEIM_mat[M_DEIM_idx[1:m], 1:m])) ≤ 1e-80
      error("Something went wrong with the construction of (M)DEIM basis:
        obtaining singular nested matrices during (M)DEIM offline phase")
    end
  end
  unique!(M_DEIM_idx)
  M_DEIM_err_bound = (Σ[min(n + 1, length(Σ))] *
                      norm(M_DEIM_mat[M_DEIM_idx, 1:n]' \ I(n)))

  M_DEIM_mat[:, 1:n], M_DEIM_idx, M_DEIM_err_bound

end

function MDEIM_offline(RBInfo::ROMInfoSteady{T}, var::String) where T

  println("Building $(RBInfo.nₛ_MDEIM) snapshots of $var")

  μ = load_CSV(Array{T}[],joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id, RBInfo.FEMInfo, model)

  MDEIM_mat, Σ, row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, μ, var)
  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(MDEIM_mat, Σ)
  MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx, :]
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx, row_idx, FEMSpace.Nₛᵘ)
  MDEIM_idx_sparse_space, _ = from_vec_to_mat_idx(MDEIM_idx_sparse, FEMSpace.Nₛᵘ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(MDEIM_idx_sparse_space))

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, row_idx, el

end

function MDEIM_offline(RBInfo::ROMInfoUnsteady{T}, var::String) where T

  println("Building $(RBInfo.nₛ_MDEIM) snapshots of $var")

  μ = load_CSV(Array{T}[],joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id, RBInfo.FEMInfo, model)

  MDEIM_mat, MDEIM_mat_time, Σ, row_idx = get_snaps_MDEIM(FEMSpace, RBInfo, μ, var)

  MDEIM_mat, MDEIM_idx, MDEIM_err_bound = M_DEIM_offline(MDEIM_mat, Σ)
  MDEIMᵢ_mat = MDEIM_mat[MDEIM_idx, :]
  MDEIM_idx_sparse = from_full_idx_to_sparse_idx(MDEIM_idx, row_idx, FEMSpace.Nₛᵘ)
  MDEIM_idx_sparse_space, _ = from_vec_to_mat_idx(MDEIM_idx_sparse, FEMSpace.Nₛᵘ)
  el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(MDEIM_idx_sparse_space))

  _, MDEIM_idx_time, _ = M_DEIM_offline(MDEIM_mat_time, Σ)
  unique!(sort!(MDEIM_idx_time))

  MDEIM_mat, MDEIM_idx_sparse, MDEIMᵢ_mat, row_idx, el, MDEIM_idx_time

end

function DEIM_offline(RBInfo::ROMInfoSteady{T}, var::String) where T

  println("Building $(RBInfo.nₛ_DEIM) snapshots of $var")

  μ = load_CSV(Array{T}[], joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id, RBInfo.FEMInfo, model)

  DEIM_mat, Σ = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)
  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(DEIM_mat, Σ)
  DEIMᵢ_mat = DEIM_mat[DEIM_idx, :]
  if var == "H"
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Γn, unique(DEIM_idx))
  else
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(DEIM_idx))
  end

  DEIM_mat, DEIM_idx, DEIMᵢ_mat, el

end

function DEIM_offline(RBInfo::ROMInfoUnsteady{T}, var::String) where T

  println("Building $(RBInfo.nₛ_DEIM) snapshots of $var")

  μ = load_CSV(Array{T}[], joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id, RBInfo.FEMInfo, model)

  DEIM_mat, DEIM_mat_time, Σ = get_snaps_DEIM(FEMSpace, RBInfo, μ, var)

  DEIM_mat, DEIM_idx, DEIM_err_bound = M_DEIM_offline(DEIM_mat, Σ)
  DEIMᵢ_mat = DEIM_mat[DEIM_idx, :]
  if var == "H"
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Γn, unique(DEIM_idx))
  else
    el = find_FE_elements(FEMSpace.V₀, FEMSpace.Ω, unique(DEIM_idx))
  end

  _, DEIM_idx_time, _ = M_DEIM_offline(DEIM_mat_time, Σ)
  unique!(sort!(DEIM_idx_time))

  DEIM_mat, DEIM_idx, DEIMᵢ_mat, el, DEIM_idx_time

end

function M_DEIM_online(Mat_nonaffine, Matᵢ::Matrix{T}, idx::Vector{Int}) where T
  @fastmath Matᵢ \ Matrix{T}(reshape(Mat_nonaffine, :, 1)[idx, :])
end
