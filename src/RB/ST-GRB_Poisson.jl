function get_Aₙ(
  RBInfo::Info,
  RBVars::PoissonSTGRB)

  get_Aₙ(RBInfo, RBVars.Steady)

end

function get_Mₙ(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTGRB{T}) where T

  if isfile(joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))
    println("Importing reduced affine mass matrix")
    Mₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))
    RBVars.Mₙ = reshape(Mₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}
    RBVars.Qᵐ = size(RBVars.Mₙ)[end]
    return [""]
  else
    println("Failed to import the reduced affine mass matrix: must build it")
    return ["M"]
  end

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::PoissonSTGRB{T},
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
  RBVars::PoissonSTGRB{T},
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
  else
    RBVars.Aₙ = Matₙ
    RBVars.Qᵃ = Q
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::PoissonSTGRB{T},
  var::String) where T

  assemble_affine_vectors(RBInfo, RBVars.Steady, var)

end

function assemble_reduced_mat_DEIM(
  RBVars::PoissonSTGRB{T},
  DEIM_mat::Matrix{T},
  var::String) where T

  Q = size(DEIM_mat)[2]
  Vecₙ = zeros(T, RBVars.nₛᵘ,1,Q)
  @simd for q = 1:Q
    Vecₙ[:,:,q] = RBVars.Φₛᵘ' * Vector{T}(DEIM_mat[:, q])
  end
  Vecₙ = reshape(Vecₙ,:,Q)::Matrix{T}

  if var == "F"
    RBVars.Fₙ = Vecₙ
    RBVars.Qᶠ = Q
  elseif var == "H"
    RBVars.Hₙ = Vecₙ
    RBVars.Qʰ = Q
  else
    error("Unrecognized vector to assemble with DEIM")
  end

end

function assemble_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTGRB,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.offline_time += @elapsed begin
    if "M" ∈ operators
      if !RBInfo.probl_nl["M"]
        assemble_affine_matrices(RBInfo, RBVars, "M")
      else
        assemble_MDEIM_matrices(RBInfo, RBVars, "M")
      end
    end

    if "A" ∈ operators
      if !RBInfo.probl_nl["A"]
        assemble_affine_matrices(RBInfo, RBVars, "A")
      else
        assemble_MDEIM_matrices(RBInfo, RBVars, "A")
      end
    end

    if "F" ∈ operators
      if !RBInfo.probl_nl["f"]
        assemble_affine_vectors(RBInfo, RBVars, "F")
      else
        assemble_DEIM_vectors(RBInfo, RBVars, "F")
      end
    end

    if "H" ∈ operators
      if !RBInfo.probl_nl["h"]
        assemble_affine_vectors(RBInfo, RBVars, "H")
      else
        assemble_DEIM_vectors(RBInfo, RBVars, "H")
      end
    end
  end

  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::PoissonSTGRB{T}) where T

  if RBInfo.save_offline_structures
    save_CSV(reshape(RBVars.Mₙ, :, RBVars.Qᵐ)::Matrix{T},
      joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))
    save_affine_structures(RBInfo, RBVars.Steady)
  end

end

function get_affine_structures(
  RBInfo::Info,
  RBVars::PoissonSTGRB)

  operators = String[]
  append!(operators, get_Mₙ(RBInfo, RBVars))
  append!(operators, get_affine_structures(RBInfo, RBVars.Steady))
  return operators

end

function get_Q(
  RBInfo::Info,
  RBVars::PoissonSTGRB)

  if RBVars.Qᵐ == 0
    RBVars.Qᵐ = size(RBVars.Mₙ)[end]
  end

  get_Q(RBInfo, RBVars.Steady)

end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTGRB{T},
  θᵐ::Matrix{T},
  θᵃ::Matrix{T}) where T

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
    Aₙ_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ_A[:,:,qᵃ])::Matrix{T}
    Aₙ₁_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ₁_A[:,:,qᵃ])::Matrix{T}
  end
  Mₙ = reshape(sum(Mₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Mₙ₁ = reshape(sum(Mₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ = δtθ*reshape(sum(Aₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ₁ = δtθ*reshape(sum(Aₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)

  block₁ = θ*(Aₙ+Mₙ) + (1-θ)*Aₙ₁ - θ*Mₙ₁
  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTGRB{T},
  θᶠ::Array{T},
  θʰ::Array{T}) where T

  println("Assembling RHS using θ-method time scheme, θ=$(RBInfo.θ)")

  Qᶠ = RBVars.Qᶠ
  Qʰ = RBVars.Qʰ
  δtθ = RBInfo.δt*RBInfo.θ
  nₜᵘ = RBVars.nₜᵘ

  Φₜᵘ_F = zeros(T, RBVars.nₜᵘ, Qᶠ)
  Φₜᵘ_H = zeros(T, RBVars.nₜᵘ, Qʰ)
  @simd for i_t = 1:nₜᵘ
    for q = 1:Qᶠ
      Φₜᵘ_F[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θᶠ[q,:])
    end
    for q = 1:Qʰ
      Φₜᵘ_H[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θʰ[q,:])
    end
  end

  block₁ = zeros(T, RBVars.nᵘ,1)
  @simd for i_s = 1:RBVars.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ
      i_st = index_mapping(i_s, i_t, RBVars)
      Fₙ_μ_i_j = RBVars.Fₙ[i_s,:]'*Φₜᵘ_F[i_t,:]
      Hₙ_μ_i_j = RBVars.Hₙ[i_s,:]'*Φₜᵘ_H[i_t,:]
      block₁[i_st,1] = Fₙ_μ_i_j+Hₙ_μ_i_j
    end
  end

  block₁ *= δtθ
  push!(RBVars.RHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTGRB,
  Param::UnsteadyParametricInfo)

  initialize_RB_system(RBVars.Steady)
  initialize_online_time(RBVars.Steady)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    blocks = [1]
    operators = get_system_blocks(RBInfo,RBVars.Steady,blocks,blocks)

    θᵐ, θᵃ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ)
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ)
      else
        build_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
      if RBInfo.probl_nl["g"]
        build_RB_lifting(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars.Steady,blocks,blocks,operators)

end

function build_RB_lifting(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTGRB{T},
  Param::UnsteadyParametricInfo) where T

  println("Assembling reduced lifting exactly")

  L_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")
  L = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo)
  for (i,tᵢ) in enumerate(timesθ)
    L[:,i] = L_t(tᵢ)
  end
  Lₙ = Matrix{T}[]
  push!(Lₙ, reshape((RBVars.Φₛᵘ'*(L*RBVars.Φₜᵘ))',:,1))::Vector{Matrix{T}}
  RBVars.RHSₙ -= Lₙ

end

function build_param_RHS(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTGRB{T},
  Param::UnsteadyParametricInfo) where T

  println("Assembling RHS exactly using θ-method time scheme, θ=$(RBInfo.θ)")
  δtθ = RBInfo.δt*RBInfo.θ
  F_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  F, H = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ), zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo)
  for (i,tᵢ) in enumerate(timesθ)
    F[:,i] = F_t(tᵢ)
    H[:,i] = H_t(tᵢ)
  end
  F *= δtθ
  H *= δtθ
  Fₙ = RBVars.Φₛᵘ'*(F*RBVars.Φₜᵘ)
  Hₙ = RBVars.Φₛᵘ'*(H*RBVars.Φₜᵘ)
  push!(RBVars.RHSₙ, reshape(Fₙ'+Hₙ',:,1))::Vector{Matrix{T}}

end

function get_θ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonSTGRB{T},
  Param::UnsteadyParametricInfo) where T

  θᵐ = get_θᵐ(FEMSpace, RBInfo, RBVars, Param)
  θᵃ = get_θᵃ(FEMSpace, RBInfo, RBVars, Param)
  if !RBInfo.build_parametric_RHS
    θᶠ, θʰ = get_θᶠʰ(FEMSpace, RBInfo, RBVars, Param)
  else
    θᶠ, θʰ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  end

  return θᵐ, θᵃ, θᶠ, θʰ

end
