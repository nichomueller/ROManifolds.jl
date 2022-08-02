function get_Aₙ(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  get_Aₙ(RBInfo, RBVars.Stokes)

end

function get_Mₙ(
  RBInfo::ROMInfoUnsteady,
  RBVars::NavierStokesSTGRB)

  get_Mₙ(RBInfo, RBVars.Stokes)

end

function get_Bₙ(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  get_Bₙ(RBInfo, RBVars.Stokes)

end

function get_Cₙ(
  RBInfo::Info,
  RBVars::NavierStokesSGRB{T}) where T

  get_Cₙ(RBInfo, RBVars.Steady)

end

function get_Fₙ(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  get_Fₙ(RBInfo, RBVars.Stokes)

end

function get_Hₙ(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  get_Hₙ(RBInfo, RBVars.Stokes)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  var::String)

  assemble_affine_matrices(RBInfo, RBVars.Steady, var)

end

function assemble_reduced_mat_MDEIM(
  RBVars::NavierStokesSTGRB,
  MDEIM_mat::Matrix,
  row_idx::Vector{Int},
  var::String)

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
  elseif var == "C"
    RBVars.Cₙ = Matₙ
    RBVars.Qᶜ = Q
  elseif var == "D"
    RBVars.Dₙ = Matₙ
    RBVars.Qᵈ = Q
  else
    error("Unrecognized variable")
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  var::String)

  assemble_affine_vectors(RBInfo, RBVars.Stokes, var)

end

function assemble_reduced_mat_DEIM(
  RBInfo::ROMInfoUnsteady,
  RBVars::NavierStokesSTGRB,
  DEIM_mat::Matrix,
  var::String)

  assemble_reduced_mat_DEIM(RBInfo, RBVars.Stokes, DEIM_mat, var)

end

function assemble_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSTGRB,
  operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assemble_offline_structures(RBInfo, RBVars.Stokes, operators)

  RBVars.offline_time += @elapsed begin
    if "C" ∈ operators
      assemble_affine_matrices(RBInfo, RBVars, "C")
    end

  end

  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  save_affine_structures(RBInfo, RBVars.Steady)

end

function get_affine_structures(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  operators = String[]
  append!(operators, get_Cₙ(RBInfo, RBVars))
  append!(operators, get_affine_structures(RBInfo, RBVars.Stokes))

  operators

end

function get_Q(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  if RBVars.Qᶜ == 0
    RBVars.Qᶜ = size(RBVars.Cₙ)[end]
  end
  get_Q(RBInfo, RBVars.Stokes)

end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::NavierStokesSTGRB{T},
  θᵐ::Matrix,
  θᵃ::Matrix,
  θᵇ::Matrix,
  θᶜ::Matrix) where T

  println("Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)")

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ
  nₜᵘ = RBVars.nₜᵘ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.Qᵃ
  Qᶜ = RBVars.Qᶜ

  Φₜᵘ_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ₁_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)
  Φₜᵘ₁_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)
  Φₜᵘ_C = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᶜ)
  Φₜᵘ₁_C = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᶜ)

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
      for q = 1:Qᶜ
        Φₜᵘ_C[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᶜ[q,:])
        Φₜᵘ₁_C[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᶜ[q,2:end])
      end
    end
  end

  Mₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Mₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Aₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)
  Aₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)
  Cₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᶜ)
  Cₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᶜ)

  @simd for qᵐ = 1:Qᵐ
    Mₙ_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ_M[:,:,qᵐ])::Matrix{T}
    Mₙ₁_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ₁_M[:,:,qᵐ])::Matrix{T}
  end
  @simd for qᵃ = 1:Qᵃ
    Aₙ_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ_A[:,:,qᵃ])::Matrix{T}
    Aₙ₁_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ₁_A[:,:,qᵃ])::Matrix{T}
  end
  @simd for qᶜ = 1:Qᶜ
    Cₙ_tmp[:,:,qᶜ] = kron(RBVars.Cₙ[:,:,qᶜ],Φₜᵘ_C[:,:,qᶜ])::Matrix{T}
    Cₙ₁_tmp[:,:,qᶜ] = kron(RBVars.Cₙ[:,:,qᶜ],Φₜᵘ₁_C[:,:,qᶜ])::Matrix{T}
  end
  Mₙ = reshape(sum(Mₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Mₙ₁ = reshape(sum(Mₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ = δtθ*reshape(sum(Aₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ₁ = δtθ*reshape(sum(Aₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Cₙ = δtθ*reshape(sum(Cₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Cₙ₁ = δtθ*reshape(sum(Cₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)

  Jₙ = Aₙ + Cₙ
  Jₙ₁ = Aₙ₁ + Cₙ₁

  block₁ = θ*(Jₙ+Mₙ) + (1-θ)*Jₙ₁ - θ*Mₙ₁

  Φₜᵘᵖ = RBVars.Φₜᵘ' * RBVars.Φₜᵖ
  Bₙᵀ = permutedims(RBVars.Bₙ,[2,1,3])::Array{T,3}
  Bₙᵀ = kron(Bₙᵀ[:,:,1].*θᵇ, Φₜᵘᵖ)::Matrix{T}
  Bₙ = (Bₙᵀ)'::Matrix{T}

  block₂ = -RBInfo.δt*RBInfo.θ * Bₙᵀ
  block₃ = Bₙ

  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, block₂)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, block₃)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, zeros(T, RBVars.nᵖ, RBVars.nᵖ))::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB{T},
  θᶠ::Matrix,
  θʰ::Matrix) where T

  get_RB_RHS_blocks(RBInfo, RBVars.Stokes, θᶠ, θʰ)

end

function get_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  Param::ParametricInfoUnsteady)

  initialize_RB_system(RBVars.Steady)
  initialize_online_time(RBVars.Steady)

  LHS_blocks = [1, 2, 3]
  RHS_blocks = [1]

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)

    operators = get_system_blocks(RBInfo,RBVars.Steady,LHS_blocks,RHS_blocks)

    θᵐ, θᵃ, θᵇ, θᶜ, θᶠ, θʰ  = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ, θᵇ, θᶜ)
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ)
      else
        build_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars.Steady,LHS_blocks,RHS_blocks,operators)

end

function build_param_RHS(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  Param::ParametricInfoUnsteady)

  build_param_RHS(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function get_θ(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  Param::ParametricInfoUnsteady)

  θᵐ, θᵃ, θᵇ, θᶠ, θʰ  = get_θ(FEMSpace, RBInfo, RBVars.Stokes, Param)
  θᶜ = get_θᶜ(FEMSpace, RBInfo, RBVars, Param)

  return θᵐ, θᵃ, θᵇ, θᶜ, θᶠ, θʰ

end
