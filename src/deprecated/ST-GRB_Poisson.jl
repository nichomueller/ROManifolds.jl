include("S-GRB_Poisson.jl")
include("RB_Poisson_unsteady.jl")

function get_RB_LHS_blocks(RBInfo, RBVars::PoissonSTGRB, θᵐ, θᵃ)

  println("Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)")

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ
  nₜᵘ = RBVars.nₜᵘ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.Qᵃ

  Φₜᵘ_M = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)
  Φₜᵘ₁_M = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)
  Φₜᵘ_A = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)
  Φₜᵘ₁_A = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐ[q,:]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵃ[q,:]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  block₁ = zeros(RBVars.nᵘ, RBVars.nᵘ)

  for i_s = 1:RBVars.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      for j_s = 1:RBVars.nₛᵘ
        for j_t = 1:RBVars.nₜᵘ

          j_st = index_mapping(j_s, j_t, RBVars)

          Aₙ_μ_i_j = δtθ*RBVars.Aₙ[i_s,j_s,:]'*Φₜᵘ_A[i_t,j_t,:]
          Mₙ_μ_i_j = RBVars.Mₙ[i_s,j_s,:]'*Φₜᵘ_M[i_t,j_t,:]
          Aₙ₁_μ_i_j = δtθ*RBVars.Aₙ[i_s,j_s,:]'*Φₜᵘ₁_A[i_t,j_t,:]
          Mₙ₁_μ_i_j = RBVars.Mₙ[i_s,j_s,:]'*Φₜᵘ₁_M[i_t,j_t,:]

          block₁[i_st,j_st] = θ*(Aₙ_μ_i_j+Mₙ_μ_i_j) + (1-θ)*Aₙ₁_μ_i_j - θ*Mₙ₁_μ_i_j

        end
      end

    end
  end

  Mₙ_μ_i_j = zeros(RBVars.nᵘ, RBVars.nᵘ)
  for i_s = 1:RBVars.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ
      i_st = index_mapping(i_s, i_t, RBVars)
      for j_s = 1:RBVars.nₛᵘ
        for j_t = 1:RBVars.nₜᵘ
          j_st = index_mapping(j_s, j_t, RBVars)
          Mₙ_μ_i_j[i_st,j_st] = RBVars.Mₙ[i_s,j_s,:]'*Φₜᵘ_M[i_t,j_t,:]
        end
      end
    end
  end

  push!(RBVars.LHSₙ, block₁)

end



function get_RB_RHS_blocks(RBInfo::Info, RBVars::PoissonSTGRB, θᶠ, θʰ)

  println("Assembling RHS using θ-method time scheme, θ=$(RBInfo.θ)")

  Qᶠ = RBVars.Qᶠ
  Qʰ = RBVars.Qʰ
  δtθ = RBInfo.δt*RBInfo.θ
  nₜᵘ = RBVars.nₜᵘ

  Φₜᵘ_F = zeros(RBVars.nₜᵘ, Qᶠ)
  Φₜᵘ_H = zeros(RBVars.nₜᵘ, Qʰ)
  [Φₜᵘ_F[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θᶠ[q,:]) for q = 1:Qᶠ for i_t = 1:nₜᵘ]
  [Φₜᵘ_H[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θʰ[q,:]) for q = 1:Qʰ for i_t = 1:nₜᵘ]

  Fₙ_tmp = zeros(RBVars.nᵘ,Qᶠ)
  [Fₙ_tmp[:,q] = kronecker(RBVars.Fₙ[:,q],Φₜᵘ_F[:,q]) for q = 1:Qᶠ]
  Fₙ = sum(Fₙ_tmp,dims=2)
  Hₙ_tmp = zeros(RBVars.nᵘ,Qʰ)
  [Hₙ_tmp[:,q] = kronecker(RBVars.Hₙ[:,q],Φₜᵘ_H[:,q]) for q = 1:Qʰ]
  Hₙ = sum(Hₙ_tmp,dims=2)

  block₁ = (Fₙ+Hₙ)*δtθ

  push!(RBVars.RHSₙ, block₁)

end
