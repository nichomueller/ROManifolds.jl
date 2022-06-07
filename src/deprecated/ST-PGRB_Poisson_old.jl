
function get_MΦ(RBInfo::Problem, RBVars::PoissonSTPGRB)

  @info "S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹"
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MΦ.csv"))
    MΦ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MΦ.csv"))
    RBVars.MΦ = reshape(MΦ,RBVars.steady_info.Nₛᵘ,RBVars.steady_info.nₛᵘ,:)
    return
  else
    if !RBInfo.probl_nl["M"]
      @info "S-PGRB: failed to build MΦ; have to assemble affine stiffness"
      assemble_affine_matrices(RBInfo, RBVars, "M")
    else
      @info "S-PGRB: failed to build MΦ; have to assemble non-affine stiffness "
      assemble_MDEIM_matrices(RBInfo, RBVars, "M")
    end
  end

end

function get_MAₙ(RBInfo::Problem, RBVars::PoissonSTPGRB)

  @info "S-PGRB: fetching the matrix MAₙ"
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MAₙ.csv"))
    @info "Importing reduced affine matrix MAₙ"
    RBVars.MAₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MAₙ.csv"))
    return
  else
    if !RBInfo.probl_nl["M"]
      @info "S-PGRB: failed to import MAₙ; have to assemble MΦ and AΦᵀPᵤ⁻¹"
      get_MΦ(RBInfo, RBVars)
      get_AΦᵀPᵤ⁻¹(RBInfo, RBVars.steady_info)
      nₛᵘ = RBVars.steady_info.nₛᵘ
      MAₙ = zeros(RBVars.steady_info.nₛᵘ,RBVars.steady_info.nₛᵘ,RBVars.Qᵐ*RBVars.Qᵃ)
      [MAₙ[:,:,(i-1)*nₛᵘ+j] = RBVars.MΦ'[:,:,i] * RBVars.steady_info.AΦᵀPᵤ⁻¹'[:,:,j] for i=1:nₛᵘ for j=1:nₛᵘ]
      RBVars.MAₙ = MAₙ
    end
  end

end

function assemble_affine_matrices(RBInfo::Problem, RBVars::PoissonSTPGRB, var::String)

  get_inverse_P_matrix(RBInfo, RBVars)

  if var === "M"
    RBVars.Qᵐ = 1
    @info "Assembling affine reduced mass"
    M = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "M.csv"); convert_to_sparse = true)
    RBVars.Mₙ = zeros(RBVars.steady_info.nₛᵘ, RBVars.steady_info.nₛᵘ, RBVars.Qᵐ)
    RBVars.Mₙ[:,:,1] = (M*RBVars.steady_info.Φₛᵘ)' * RBVars.Pᵤ⁻¹ * (M*RBVars.steady_info.Φₛᵘ)
    RBVars.MΦ = zeros(RBVars.steady_info.Nₛᵘ, RBVars.steady_info.nₛᵘ, RBVars.Qᵐ)
    RBVars.MΦ[:,:,1] = M*RBVars.steady_info.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.steady_info, var)
  end

end

function assemble_MDEIM_matrices(RBInfo::Problem, RBVars::PoissonSTPGRB, var::String)

  get_inverse_P_matrix(RBInfo, RBVars)

  if var === "M"

    @info "The mass is non-affine: running the MDEIM offline phase on $nₛ_MDEIM snapshots. This might take some time"
    MDEIM_mat, RBVars.MDEIM_idx_M, RBVars.sparse_el_M, _, _ = MDEIM_offline(FEMInfo, RBInfo, "M")
    RBVars.Qᵐ = size(MDEIM_mat)[2]

    MΦP_inv = zeros(RBVars.steady_info.nₛᵘ, RBVars.steady_info.Nₛᵘ, RBVars.Qᵐ)
    RBVars.Mₙ = zeros(RBVars.steady_info.nₛᵘ, RBVars.steady_info.nₛᵘ, RBVars.Qᵐ^2)
    RBVars.MΦ = zeros(RBVars.steady_info.Nₛᵘ, RBVars.steady_info.nₛᵘ, RBVars.Qᵐ)
    for q = 1:RBVars.Qᵐ
      RBVars.MΦ[:,:,q] = reshape(Vector(MDEIM_mat[:, q]), RBVars.steady_info.Nₛᵘ, RBVars.steady_info.nₛᵘ) * RBVars.steady_info.Φₛᵘ
    end
    tensor_product(MΦP_inv, MΦ, RBVars.Pᵤ⁻¹, transpose_A=true)

    for q₁ = 1:RBVars.Qᵐ
      for q₂ = 1:RBVars.Qᵐ
        @info "SPGRB: affine component number $((q₁-1)*RBVars.Qᵐ+q₂), matrix M"
        RBVars.Mₙ[:, :, (q₁-1)*RBVars.Qᵐ+q₂] = MΦP_inv[:, :, q₁] * RBVars.MΦ[:, :, q₂]
      end
    end
    RBVars.MDEIMᵢ_M = Matrix(MDEIM_mat[RBVars.MDEIM_idx_M, :])

  else

    assemble_MDEIM_matrices(RBInfo, RBVars.steady_info, var)

  end

end

function assemble_affine_vectors(RBInfo::Problem, RBVars::PoissonSTPGRB, var::String)

  assemble_affine_vectors(RBInfo, RBVars.steady_info, var)

end

function assemble_DEIM_vectors(RBInfo::Problem, RBVars::PoissonSTPGRB, var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.steady_info, var)

end

function assemble_offline_structures(RBInfo::Problem, RBVars::PoissonSTPGRB, operators=nothing)

  if isnothing(operators)
    operatorsₜ = set_operators(RBInfo, RBVars)
  end

  assembly_time = 0
  if "M" ∈ operatorsₜ
    if !RBInfo.probl_nl["M"]
      assembly_time += @elapsed begin
        assemble_affine_matrices(RBInfo, RBVars, "M")
      end
    else
      assembly_time += @elapsed begin
        assemble_MDEIM_matrices(RBInfo, RBVars, "M")
      end
    end
  end

  assemble_offline_structures(RBInfo, RBVars.steady_info, operators)
  assemble_affine_matrices(RBInfo, RBVars, "MA")

  RBVars.steady_info.offline_time += assembly_time
  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function save_affine_structures(RBInfo::Problem, RBVars::PoissonSTPGRB)

  if RBInfo.save_offline_structures
    save_CSV(Mₙ, joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    save_CSV([RBVars.Qᵐ], joinpath(RBInfo.paths.ROM_structures_path, "Qᵐ.csv"))
    save_CSV(MAₙ, joinpath(RBInfo.paths.ROM_structures_path, "MAₙ.csv"))
  end

end

function get_affine_structures(RBInfo::Problem, RBVars::PoissonSTPGRB)

  operators = []

  push!(operators, get_Mₙ(RBInfo, RBVars))
  push!(operators, get_MAₙ(RBInfo, RBVars))
  push!(operators, get_affine_structures(RBInfo, RBVars.steady_info))

  operators

end

function save_affine_structures(RBInfo::Problem, RBVars::PoissonSTPGRB)

  if RBInfo.save_offline_structures
    save_CSV(Mₙ, joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    save_CSV([RBVars.Qᵐ], joinpath(RBInfo.paths.ROM_structures_path, "Qᵐ.csv"))
  end

end

function get_RB_LHS_blocks(RBInfo::Problem, RBVars::PoissonSTPGRB, Mₙ, Aₙ, MAₙ)

  @info "Assembling LHS using Crank-Nicolson time scheme"

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ

  Mat_Mat = Mₙ+(δtθ)^2*Aₙ+(δtθ)*MAₙ+(δtθ)*MAₙ'
  M_Mat = Mₙ+(δtθ)*MAₙ
  M_Mat_Mat_M = M_Mat+M_Mat'

  Φₜᵘ₁ = RBVars.Φₜᵘ[1:end-1, :]' * RBVars.Φₜᵘ[1:end-1, :]
  Φₜᵘ₁₂ = RBVars.Φₜᵘ[1:end-1, :]'*RBVars.Φₜᵘ[2:end, :]

  block1 = zeros(RBVars.nᵘ, RBVars.nᵘ)
  for i_s = 1:RBVars.steady_info.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      for j_s = 1:RBVars.steady_info.nₛᵘ
        for j_t = 1:RBVars.nₜᵘ
          j_st = index_mapping(j_s, j_t, RBVars)
          block1[i_st, j_st] += θ^2*Mat_Mat[i_s,j_s]*(i_t===j_t) + (1-θ)^2*Mat_Mat[i_s,j_s]*Φₜᵘ₁[i_t,j_t] - (1-θ)*M_Mat_Mat_M[i_s,j_s]*Φₜᵘ₁[i_t,j_t] + θ*(1-θ)*M_Mat_Mat_M[i_s,j_s]*Φₜᵘ₁₂[i_t,j_t] + θ*(1-θ)*Mat_Mat[i_s,j_s]*Φₜᵘ₁₂[j_t,i_t] + θ*Mat_Mat[j_s,i_s]*Φₜᵘ₁₂[j_t,i_t] + Mₙ*Φₜᵘ₁[i_t,j_t]
        end
      end

    end
  end

  push!(RBVars.steady_info.LHSₙ, block1)

end

#= function get_RB_RHS_blocks(RBInfo, RBVars::PoissonSTPGRB, Param)

  @info "Assembling RHS"

  Ffun = assemble_forcing(FEMSpace, Param)
  F_mat = zeros(RBVars.steady_info.Nₛᵘ, RBVars.Nₜ + 1)
  for (i, tᵢ) in enumerate(RBInfo.t₀:RBInfo.δt:RBInfo.T)
    F_mat[:, i] = Ffun(tᵢ)
  end
  F = (F_mat[:, 2:end] + F_mat[:, 1:end-1])*RBInfo.δt/2
  Fₙ = (RBVars.steady_info.Φₛᵘ)' * (F * RBVars.Φₜᵘ)
  push!(RBVars.steady_info.RHSₙ, reshape(Fₙ', :, 1))

end =#

function get_RB_RHS_blocks(RBInfo::Problem, RBVars::PoissonSTPGRB, Fₙ, Hₙ)

  @info "Assembling RHS"

  δtθ = RBInfo.δt*RBInfo.θ
  FHₙ = δtθ*(Fₙ+Hₙ)
  push!(RBVars.steady_info.RHSₙ, reshape(FHₙ', :, 1))

end

function get_RB_system(RBInfo::Problem, RBVars::PoissonSTPGRB, Param)

  @info "Preparing the RB system: fetching reduced LHS"
  initialize_RB_system(RBVars)
  θᵐ, θᵐᵃ, θᵃ, θᶠ = get_θ(RBInfo, RBVars, Param)
  blocks = [1]
  operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

  if "LHS" ∈ operators
    Mₙ = assemble_online_structure(θᵐ, RBVars.Mₙ)
    Aₙ = assemble_online_structure(θᵃ, RBVars.steady_info.Aₙ)
    MAₙ = assemble_online_structure(θᵐᵃ, RBVars.MAₙ)
    get_RB_LHS_blocks(RBInfo, RBVars, Mₙ, Aₙ, MAₙ)
  end

  if "RHS" ∈ operators
    if !RBInfo.build_Parametric_RHS
      @info "Preparing the RB system: fetching reduced RHS"
      Fₙ_μ = assemble_online_structure(θᶠ, RBVars.steady_info.Fₙ)
      Hₙ_μ = assemble_online_structure(θʰ, RBVars.steady_info.Hₙ)
      push!(RBVars.steady_info.RHSₙ, reshape(Fₙ_μ+Hₙ_μ,:,1))
    else
      @info "Preparing the RB system: assembling reduced RHS exactly"
      Fₙ_μ, Hₙ_μ = build_Param_RHS(RBInfo, RBVars, Param)
      push!(RBVars.steady_info.RHSₙ, reshape(Fₙ_μ+Hₙ_μ,:,1))
    end
  end

end

function build_Param_RHS(RBInfo::Problem, RBVars::PoissonSTPGRB, Param)

  δtθ = RBInfo.δt*RBInfo.θ

  FEMSpace = get_FEMSpace(FEMInfo, Param.model)
  F_t, H_t = assemble_forcing(FEMSpace, RBVars, Param)
  F, H = zeros(RBVars.steady_info.Nₛᵘ, RBVars.Nₜ), zeros(RBVars.steady_info.Nₛᵘ, RBVars.Nₜ)
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+δtθ
  for (i, tᵢ) in enumerate(timesθ)
    F[:,i] = F_t(tᵢ)
    H[:,i] = H_t(tᵢ)
  end
  F *= δtθ
  H *= δtθ

  θᵃ_temp = get_θᵃ(RBInfo, RBVars.steady_info, Param)
  AΦᵀPᵤ⁻¹ = assemble_online_structure(θᵃ_temp, RBVars.steady_info.AΦᵀPᵤ⁻¹)

  Fₙ = AΦᵀPᵤ⁻¹*(F*RBVars.Φₜᵘ)
  Hₙ = AΦᵀPᵤ⁻¹*(H*RBVars.Φₜᵘ)

  reshape(Fₙ, :, 1), reshape(Hₙ, :, 1)

end

function get_θ(RBInfo::Problem, RBVars::PoissonSTPGRB, Param)

  θᵐ_temp = get_θᵐ(RBInfo, RBVars, Param)
  θᵃ_temp = get_θᵃ(RBInfo, RBVars, Param)

  Qᵐ, Qᵃ = length(θᵐ_temp), length(θᵃ_temp)
  θᵐ = [θᵐ_temp[q₁]*θᵐ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qᵐ]
  θᵐᵃ = [θᵐ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qᵃ]
  θᵃᵐ = [θᵐ_temp[q₁]*θᵃ_temp[q₂] for q₂ = 1:Qᵃ for q₁ = 1:Qᵐ]

  θᵃ_temp = get_θᵃ(RBInfo, RBVars, Param)
  θᵃ = [θᵃ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qᵃ]

  if !RBInfo.build_Parametric_RHS

    θᶠ_temp, θʰ_temp = get_θᶠʰ(RBInfo, RBVars, Param)
    Qᶠ, Qʰ = length(θᶠ_temp), length(θʰ_temp)
    θᶠ = [θᵃ_temp[q₁]*θᶠ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qᶠ]
    θʰ = [θᵃ_temp[q₁]*θʰ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qʰ]

  else

    θᶠ, θʰ = Float64[], Float64[]

  end

  return θᵐ, θᵐᵃ, θᵃᵐ, θᵃ, θᶠ, θʰ

end
