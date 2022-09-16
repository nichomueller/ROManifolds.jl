function get_Aₙ(
  RBInfo::Info,
  RBVars::ADRSTGRB)

  get_Aₙ(RBInfo, RBVars.Steady)

end

function get_Bₙ(
  RBInfo::Info,
  RBVars::ADRSTGRB)

  get_Bₙ(RBInfo, RBVars.Steady)

end

function get_Dₙ(
  RBInfo::Info,
  RBVars::ADRSTGRB)

  get_Dₙ(RBInfo, RBVars.Steady)

end

function get_Mₙ(
  RBInfo::ROMInfoST,
  RBVars::ADRSTGRB)

  get_Mₙ(RBInfo, RBVars.Poisson)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::ADRSTGRB{T},
  var::String) where T

  if var == "M"
    RBVars.Qᵐ = 1
    println("Assembling affine reduced mass")
    M = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "M.csv"))
    RBVars.Mₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
    RBVars.Mₙ[:,:,1] = (RBVars.Φₛᵘ)' * M * RBVars.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.Steady, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::ADRSTGRB{T},
  MDEIM_mat::Matrix{T},
  row_idx::Vector{Int},
  var::String) where T

  Q = size(MDEIM_mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.Nₛᵘ)
  MatqΦ = zeros(T, RBVars.Nₛᵘ,RBVars.nₛᵘ,Q)
  @simd for j = 1:RBVars.Nₛᵘ
    Mat_idx = findall(x -> x == j, r_idx)
    MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
  end
  Matₙ = reshape(RBVars.Φₛᵘ' *
    reshape(MatqΦ,RBVars.Nₛᵘ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}

  if var == "M"
    RBVars.Mₙ = Matₙ
    RBVars.Qᵐ = Q
  elseif var == "A"
    RBVars.Aₙ = Matₙ
    RBVars.Qᵃ = Q
  elseif var == "B"
    RBVars.Bₙ = Matₙ
    RBVars.Qᵇ = Q
  elseif var == "D"
    RBVars.Dₙ = Matₙ
    RBVars.Qᵈ = Q
  else
    error("Unrecognized variable")
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::ADRSTGRB{T},
  var::String) where T

  assemble_affine_vectors(RBInfo, RBVars.Steady, var)

end

function assemble_reduced_mat_DEIM(
  RBVars::ADRSTGRB{T},
  DEIM_mat::Matrix{T},
  var::String) where T

  assemble_reduced_mat_DEIM(RBVars.Poisson, DEIM_mat, var)

end

function assemble_offline_structures(
  RBInfo::ROMInfoST,
  RBVars::ADRSTGRB,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assemble_offline_structures(RBInfo, RBVars.Poisson, operators)

  RBVars.offline_time += @elapsed begin
    for var ∈ intersect(operators, RBInfo.probl_nl)
      assemble_MDEIM_matrices(RBInfo, RBVars, var)
    end

    for var ∈ setdiff(operators, RBInfo.probl_nl)
      assemble_affine_matrices(RBInfo, RBVars, var)
    end
  end

  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::ADRSTGRB{T}) where T

  if RBInfo.save_offline_structures
    save_CSV(reshape(RBVars.Mₙ, :, RBVars.Qᵐ)::Matrix{T},
      joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))
    save_affine_structures(RBInfo, RBVars.Steady)
  end

end

function get_affine_structures(
  RBInfo::Info,
  RBVars::ADRSTGRB)

  operators = get_affine_structures(RBInfo, RBVars.Steady)

  append!(operators, get_Mₙ(RBInfo, RBVars))

  operators

end

function get_Q(
  RBInfo::Info,
  RBVars::ADRSTGRB)

  if RBVars.Qᵐ == 0
    RBVars.Qᵐ = size(RBVars.Mₙ)[end]
  end

  get_Q(RBInfo, RBVars.Steady)

end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoST,
  RBVars::ADRSTGRB{T},
  θᵐ::Matrix{T},
  θᵃ::Matrix{T},
  θᵇ::Matrix{T},
  θᵈ::Matrix{T}) where T

  println("Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)")

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ
  nₜᵘ = RBVars.nₜᵘ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.Qᵃ
  Qᵇ = RBVars.Qᵇ
  Qᵈ = RBVars.Qᵈ

  Φₜᵘ_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ₁_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)
  Φₜᵘ₁_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)
  Φₜᵘ_B = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵇ)
  Φₜᵘ₁_B = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵇ)
  Φₜᵘ_D = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵈ)
  Φₜᵘ₁_D = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵈ)

  @simd for i_t = 1:nₜᵘ
    for j_t = 1:nₜᵘ
      for q = 1:Qᵐ
        Φₜᵘ_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐ[q,:])
        Φₜᵘ₁_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end])
      end
      for q = 1:Qᵃ
        Φₜᵘ_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵃ[q,:])
        Φₜᵘ₁_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end])
      end
      for q = 1:Qᵇ
        Φₜᵘ_B[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵇ[q,:])
        Φₜᵘ₁_B[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵇ[q,2:end])
      end
      for q = 1:Qᵈ
        Φₜᵘ_D[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵈ[q,:])
        Φₜᵘ₁_D[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵈ[q,2:end])
      end
    end
  end

  Mₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Mₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Aₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)
  Aₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)
  Bₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵇ)
  Bₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵇ)
  Dₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵈ)
  Dₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵈ)

  @simd for qᵐ = 1:Qᵐ
    Mₙ_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ_M[:,:,qᵐ])::Matrix{T}
    Mₙ₁_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ₁_M[:,:,qᵐ])::Matrix{T}
  end
  @simd for qᵃ = 1:Qᵃ
    Aₙ_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ_A[:,:,qᵃ])::Matrix{T}
    Aₙ₁_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ₁_A[:,:,qᵃ])::Matrix{T}
  end
  @simd for qᵇ = 1:Qᵇ
    Bₙ_tmp[:,:,qᵇ] = kron(RBVars.Bₙ[:,:,qᵇ],Φₜᵘ_B[:,:,qᵇ])::Matrix{T}
    Bₙ₁_tmp[:,:,qᵇ] = kron(RBVars.Bₙ[:,:,qᵇ],Φₜᵘ₁_B[:,:,qᵇ])::Matrix{T}
  end
  @simd for qᵈ = 1:Qᵈ
    Dₙ_tmp[:,:,qᵈ] = kron(RBVars.Dₙ[:,:,qᵈ],Φₜᵘ_D[:,:,qᵈ])::Matrix{T}
    Dₙ₁_tmp[:,:,qᵈ] = kron(RBVars.Dₙ[:,:,qᵈ],Φₜᵘ₁_D[:,:,qᵈ])::Matrix{T}
  end
  Mₙ = reshape(sum(Mₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Mₙ₁ = reshape(sum(Mₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ = δtθ*reshape(sum(Aₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ₁ = δtθ*reshape(sum(Aₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Bₙ = δtθ*reshape(sum(Bₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Bₙ₁ = δtθ*reshape(sum(Bₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Dₙ = δtθ*reshape(sum(Dₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Dₙ₁ = δtθ*reshape(sum(Dₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)

  Jₙ = Aₙ + Bₙ + Dₙ
  Jₙ₁ = Aₙ₁ + Bₙ₁ + Dₙ₁

  block₁ = θ*(Jₙ+Mₙ) + (1-θ)*Jₙ₁ - θ*Mₙ₁
  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBInfo::ROMInfoST,
  RBVars::ADRSTGRB{T},
  θᶠ::Array{T},
  θʰ::Array{T}) where T

  get_RB_RHS_blocks(RBInfo, RBVars.Poisson, θᶠ, θʰ)

end

function get_RB_system(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::ADRSTGRB,
  Param::ParamInfoST)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    blocks = [1]
    operators = get_system_blocks(RBInfo,RBVars,blocks,blocks)

    θᵐ, θᵃ, θᵇ, θᵈ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ, θᵇ, θᵈ,)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_RHS
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ)
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,blocks,blocks,operators)

end

function assemble_param_RHS(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::ADRSTGRB{T},
  Param::ParamInfoST) where T

  assemble_param_RHS(FEMSpace, RBInfo, RBVars.Poisson, Param)

end

function get_θ(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::ADRSTGRB{T},
  Param::ParamInfoST) where T

  θᵐ, θᵃ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars.Poisson, Param)
  θᵇ = get_θᵇ(FEMSpace, RBInfo, RBVars, Param)
  θᵈ = get_θᵈ(FEMSpace, RBInfo, RBVars, Param)

  return θᵐ, θᵃ, θᵇ, θᵈ, θᶠ, θʰ

end
