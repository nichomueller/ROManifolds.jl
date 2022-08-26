function basisA(
  RBVars::PoissonUnsteady{T},
  MDEIM_mat_time::Matrix{T}) where T

  Πᵃ = kron(RBVars.MDEIM_mat_A, MDEIM_mat_time)::Matrix{T}
  Πᵃ

end

function spacetime_MDEIM_idx(RBVars::PoissonUnsteady)
  idx = Int[]
  idx_space = RBVars.MDEIM_idx_A
  idx_time = RBVars.MDEIM_idx_time_A
  for i = 1:length(idx_time)
    append!(idx, (idx_time[i]-1)*RBVars.Nₛᵘ .+ idx_space)
  end
  idx
end

function get_θᵃ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady,
  MDEIM_mat_time::Matrix{T}) where T

  timesθ = get_timesθ(RBInfo)
  Πᵃ = basisA(RBVars, MDEIM_mat_time)
  st_idx = spacetime_MDEIM_idx(RBVars)

  A_μ_sparse = build_sparse_mat(
    FEMSpace,FEMInfo,Param,RBVars.sparse_el_A,timesθ;var="A")[:]
  θᵃ = Πᵃ[st_idx, :] \ A_μ_sparse[st_idx]

  θᵃ

end


function get_RB_LHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTGRB{T},
  θᵐ::Matrix{T},
  θᵃ::Matrix{T},
  MDEIM_mat_time::Matrix{T}) where T

  println("Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)")

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ
  nₜᵘ = RBVars.nₜᵘ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.Qᵃ

  Φₜᵘ_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ₁_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)
  Φₜᵘ₁_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)

  Πᵃ = basisA(RBVars, MDEIM_mat_time)

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
    end
  end

  Mₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Mₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Aₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)
  Aₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)

  @simd for qᵐ = 1:Qᵐ
    Mₙ_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ_M[:,:,qᵐ])::Matrix{T}
    Mₙ₁_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ₁_M[:,:,qᵐ])::Matrix{T}
  end
  @simd for qᵃ = 1:Qᵃ
    for i = 1: RBVars.Nₜ
      Aₙ_tmp[:,:,qᵃ] = RBVars.Φₜᵘ' * Πᵃ[(i-1)*RBVars.Nₛᵘ^2+1:i*RBVars.Nₛᵘ] * RBVars.Φₜᵘ
      Aₙ₁_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ₁_A[:,:,qᵃ])::Matrix{T}
    end
  end
  Mₙ = reshape(sum(Mₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Mₙ₁ = reshape(sum(Mₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ = δtθ*reshape(sum(Aₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ₁ = δtθ*reshape(sum(Aₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)

  block₁ = θ*(Aₙ+Mₙ) + (1-θ)*Aₙ₁ - θ*Mₙ₁
  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}

end
