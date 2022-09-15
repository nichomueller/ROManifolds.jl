function get_A(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_A(RBInfo, RBVars.Poisson)

end

function get_M(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB)

  get_M(RBInfo, RBVars.Poisson)

end

function get_B(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_B(RBInfo, RBVars.Steady)

end

function get_F(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_F(RBInfo, RBVars.Poisson)

end

function get_H(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_H(RBInfo, RBVars.Poisson)

end

function get_L(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_L(RBInfo, RBVars.Poisson)

end

function get_Lc(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Lc(RBInfo, RBVars.Steady)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::StokesSTGRB{T},
  var::String) where T

  if var == "B"
    println("Assembling affine primal operator B")
    B = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))
    RBVars.Bₙ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ, 1)
    RBVars.Bₙ[:,:,1] = (RBVars.Φₛᵖ)' * B * RBVars.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesSTGRB,
  MDEIM_mat::Matrix,
  row_idx::Vector{Int},
  var::String)

  if var == "B"
    Q = size(MDEIM_mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.Nₛᵖ)
    MatqΦ = zeros(T,RBVars.Nₛᵖ,RBVars.nₛᵘ,Q)
    @simd for j = 1:RBVars.Nₛᵖ
      Mat_idx = findall(x -> x == j, r_idx)
      MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
    end

    Matₙ = reshape(RBVars.Φₛᵖ' *
      reshape(MatqΦ,RBVars.Nₛᵖ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}
    RBVars.Bₙ = Matₙ
    RBVars.Qᵇ = Q

  else
    assemble_reduced_mat_MDEIM(RBVars.Poisson, MDEIM_mat, row_idx, var)
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::StokesSTGRB,
  var::String)

  assemble_affine_vectors(RBInfo, RBVars.Steady, var)

end

function assemble_reduced_mat_DEIM(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB,
  DEIM_mat::Matrix,
  var::String)

  assemble_reduced_mat_DEIM(RBInfo, RBVars.Steady, DEIM_mat, var)

end

function assemble_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.offline_time += @elapsed begin
    for var ∈ intersect(operators, RBInfo.probl_nl)
      if var ∈ ("A", "B", "M")
        assemble_MDEIM_matrices(RBInfo, RBVars, var)
      else
        assemble_DEIM_vectors(RBInfo, RBVars, var)
      end
    end

    for var ∈ setdiff(operators, RBInfo.probl_nl)
      if var ∈ ("A", "B", "M")
        assemble_affine_matrices(RBInfo, RBVars, var)
      else
        assemble_affine_vectors(RBInfo, RBVars, var)
      end
    end
  end

  save_assembled_structures(RBInfo, RBVars)

end

function get_Q(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  if RBVars.Qᵇ == 0
    RBVars.Qᵇ = size(RBVars.Bₙ)[end]
  end
  if !RBInfo.build_parametric_RHS
    if RBVars.Qˡᶜ == 0
      RBVars.Qˡᶜ = size(RBVars.Lcₙ)[end]
    end
  end

  get_Q(RBInfo, RBVars.Poisson)

end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB{T},
  θᵐ::Matrix,
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  get_RB_LHS_blocks(RBInfo, RBVars.Poisson, θᵐ, θᵃ)

  Qᵇ = RBVars.Qᵇ
  Φₜᵖᵘ_B = zeros(T, RBVars.nₜᵖ, RBVars.nₜᵘ, Qᵇ)

  @simd for i_t = 1:nₜᵖ
    for j_t = 1:nₜᵘ
      for q = 1:Qᵇ
        Φₜᵖᵘ_B[i_t,j_t,q] = sum(RBVars.Φₜᵖ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐ[q,:])
      end
    end
  end

  Bₙ_tmp = zeros(T, RBVars.nᵖ, RBVars.nᵘ, Qᵇ)

  @simd for qᵇ = 1:Qᵇ
    Bₙ_tmp[:,:,qᵇ] = kron(RBVars.Bₙ[:,:,qᵇ], Φₜᵖᵘ_B[:,:,qᵇ])::Matrix{T}
  end
  Bₙ = reshape(sum(Bₙ_tmp, dims=3), RBVars.nᵖ, RBVars.nᵘ)
  Bₙᵀ = Matrix(Bₙ')

  block₂ = -RBInfo.δt*RBInfo.θ * Bₙᵀ
  block₃ = Bₙ

  push!(RBVars.LHSₙ, block₂)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, block₃)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBInfo::Info,
  RBVars::StokesSTGRB{T},
  θᶠ::Matrix,
  θʰ::Matrix,
  θˡ::Matrix,
  θˡᶜ::Matrix,) where T

  println("Assembling RHS")

  get_RB_RHS_blocks(RBInfo, RBVars.Poisson, θᶠ, θʰ, θˡ)

  Φₜᵖ_Lc = zeros(T, RBVars.nₜᵖ, RBVars.Qˡᶜ)
  @simd for i_t = 1:RBVars.nₜᵖ
    for q = 1:RBVars.Qˡᶜ
      Φₜᵖ_Lc[i_t, q] = sum(RBVars.Φₜᵖ[:, i_t].*θˡᶜ[q,:])
    end
  end
  block₂ = zeros(T, RBVars.nᵖ, 1)
  @simd for i_s = 1:RBVars.nₛᵖ
    for i_t = 1:RBVars.nₜᵖ
      i_st = index_mapping(i_s, i_t, RBVars, "p")
      block₂[i_st, :] = - RBVars.Lcₙ[i_s,:]' * Φₜᵘ_Lc[i_t,:]
    end
  end

  push!(RBVars.RHSₙ, block₂)

end

function get_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::StokesSTGRB,
  Param::UnsteadyParametricInfo)

  initialize_RB_system(RBVars.Steady)
  initialize_online_time(RBVars.Steady)

  LHS_blocks = [1, 2, 3]
  RHS_blocks = [1, 2]

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)

    operators = get_system_blocks(RBInfo,RBVars.Steady,LHS_blocks,RHS_blocks)

    θᵃ, θᵇ, θᵐ, θᶠ, θʰ, θˡ, θˡᶜ  = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ, θᵇ)
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ, θˡ, θˡᶜ)
      else
        build_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo, RBVars, LHS_blocks, RHS_blocks, operators)

end

function build_param_RHS(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::StokesSTGRB,
  Param::UnsteadyParametricInfo)

  build_param_RHS(FEMSpace, RBInfo, RBVars.Poisson, Param)

  Lc_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "Lc")

  RHS_c = zeros(T, RBVars.Nₛᵖ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo)

  for (i,tᵢ) in enumerate(timesθ)
    RHS_c[:, i] = - Lc_t(tᵢ)
  end

  RHS_cₙ = RBVars.Φₛᵘ' * (RHS_c * RBVars.Φₜᵘ)
  push!(RBVars.RHSₙ, reshape(RHS_cₙ', :, 1))::Vector{Matrix{T}}

end

function get_θ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB{T},
  Param::UnsteadyParametricInfo) where T

  θᵃ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "A")
  θᵇ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "B")
  θᵐ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "M")

  if !RBInfo.build_parametric_RHS
    θᶠ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "F")
    θʰ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "H")
    θˡ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "L")
    θˡᶜ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "Lc")
  else
    θᶠ, θʰ, θˡ, θˡᶜ = (Matrix{T}(undef,0,0), Matrix{T}(undef,0,0),
      Matrix{T}(undef,0,0), Matrix{T}(undef,0,0))
  end

  return θᵃ, θᵇ, θᵐ, θᶠ, θʰ, θˡ, θˡᶜ

end
