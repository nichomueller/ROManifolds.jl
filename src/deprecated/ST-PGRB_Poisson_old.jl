
function get_MΦ(ROM_info::Problem, RB_variables::PoissonSTPGRB)

  @info "S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹"
  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "MΦ.csv"))
    MΦ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MΦ.csv"))
    RB_variables.MΦ = reshape(MΦ,RB_variables.steady_info.Nₛᵘ,RB_variables.steady_info.nₛᵘ,:)
    return
  else
    if !ROM_info.probl_nl["M"]
      @info "S-PGRB: failed to build MΦ; have to assemble affine stiffness"
      assemble_affine_matrices(ROM_info, RB_variables, "M")
    else
      @info "S-PGRB: failed to build MΦ; have to assemble non-affine stiffness "
      assemble_MDEIM_matrices(ROM_info, RB_variables, "M")
    end
  end

end

function get_MAₙ(ROM_info::Problem, RB_variables::PoissonSTPGRB)

  @info "S-PGRB: fetching the matrix MAₙ"
  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "MAₙ.csv"))
    @info "Importing reduced affine matrix MAₙ"
    RB_variables.MAₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MAₙ.csv"))
    return
  else
    if !ROM_info.probl_nl["M"]
      @info "S-PGRB: failed to import MAₙ; have to assemble MΦ and AΦᵀPᵤ⁻¹"
      get_MΦ(ROM_info, RB_variables)
      get_AΦᵀPᵤ⁻¹(ROM_info, RB_variables.steady_info)
      nₛᵘ = RB_variables.steady_info.nₛᵘ
      MAₙ = zeros(RB_variables.steady_info.nₛᵘ,RB_variables.steady_info.nₛᵘ,RB_variables.Qᵐ*RB_variables.Qᵃ)
      [MAₙ[:,:,(i-1)*nₛᵘ+j] = RB_variables.MΦ'[:,:,i] * RB_variables.steady_info.AΦᵀPᵤ⁻¹'[:,:,j] for i=1:nₛᵘ for j=1:nₛᵘ]
      RB_variables.MAₙ = MAₙ
    end
  end

end

function assemble_affine_matrices(ROM_info::Problem, RB_variables::PoissonSTPGRB, var::String)

  get_inverse_P_matrix(ROM_info, RB_variables)

  if var === "M"
    RB_variables.Qᵐ = 1
    @info "Assembling affine reduced mass"
    M = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "M.csv"); convert_to_sparse = true)
    RB_variables.Mₙ = zeros(RB_variables.steady_info.nₛᵘ, RB_variables.steady_info.nₛᵘ, RB_variables.Qᵐ)
    RB_variables.Mₙ[:,:,1] = (M*RB_variables.steady_info.Φₛᵘ)' * RB_variables.Pᵤ⁻¹ * (M*RB_variables.steady_info.Φₛᵘ)
    RB_variables.MΦ = zeros(RB_variables.steady_info.Nₛᵘ, RB_variables.steady_info.nₛᵘ, RB_variables.Qᵐ)
    RB_variables.MΦ[:,:,1] = M*RB_variables.steady_info.Φₛᵘ
  else
    assemble_affine_matrices(ROM_info, RB_variables.steady_info, var)
  end

end

function assemble_MDEIM_matrices(ROM_info::Problem, RB_variables::PoissonSTPGRB, var::String)

  get_inverse_P_matrix(ROM_info, RB_variables)

  if var === "M"

    @info "The mass is non-affine: running the MDEIM offline phase on $nₛ_MDEIM snapshots. This might take some time"
    MDEIM_mat, RB_variables.MDEIM_idx_M, RB_variables.sparse_el_M, _, _ = MDEIM_offline(problem_info, ROM_info, "M")
    RB_variables.Qᵐ = size(MDEIM_mat)[2]

    MΦP_inv = zeros(RB_variables.steady_info.nₛᵘ, RB_variables.steady_info.Nₛᵘ, RB_variables.Qᵐ)
    RB_variables.Mₙ = zeros(RB_variables.steady_info.nₛᵘ, RB_variables.steady_info.nₛᵘ, RB_variables.Qᵐ^2)
    RB_variables.MΦ = zeros(RB_variables.steady_info.Nₛᵘ, RB_variables.steady_info.nₛᵘ, RB_variables.Qᵐ)
    for q = 1:RB_variables.Qᵐ
      RB_variables.MΦ[:,:,q] = reshape(Vector(MDEIM_mat[:, q]), RB_variables.steady_info.Nₛᵘ, RB_variables.steady_info.nₛᵘ) * RB_variables.steady_info.Φₛᵘ
    end
    tensor_product(MΦP_inv, MΦ, RB_variables.Pᵤ⁻¹, transpose_A=true)

    for q₁ = 1:RB_variables.Qᵐ
      for q₂ = 1:RB_variables.Qᵐ
        @info "SPGRB: affine component number $((q₁-1)*RB_variables.Qᵐ+q₂), matrix M"
        RB_variables.Mₙ[:, :, (q₁-1)*RB_variables.Qᵐ+q₂] = MΦP_inv[:, :, q₁] * RB_variables.MΦ[:, :, q₂]
      end
    end
    RB_variables.MDEIMᵢ_M = Matrix(MDEIM_mat[RB_variables.MDEIM_idx_M, :])

  else

    assemble_MDEIM_matrices(ROM_info, RB_variables.steady_info, var)

  end

end

function assemble_affine_vectors(ROM_info::Problem, RB_variables::PoissonSTPGRB, var::String)

  assemble_affine_vectors(ROM_info, RB_variables.steady_info, var)

end

function assemble_DEIM_vectors(ROM_info::Problem, RB_variables::PoissonSTPGRB, var::String)

  assemble_DEIM_vectors(ROM_info, RB_variables.steady_info, var)

end

function assemble_offline_structures(ROM_info::Problem, RB_variables::PoissonSTPGRB, operators=nothing)

  if isnothing(operators)
    operatorsₜ = set_operators(ROM_info, RB_variables)
  end

  assembly_time = 0
  if "M" ∈ operatorsₜ
    if !ROM_info.probl_nl["M"]
      assembly_time += @elapsed begin
        assemble_affine_matrices(ROM_info, RB_variables, "M")
      end
    else
      assembly_time += @elapsed begin
        assemble_MDEIM_matrices(ROM_info, RB_variables, "M")
      end
    end
  end

  assemble_offline_structures(ROM_info, RB_variables.steady_info, operators)
  assemble_affine_matrices(ROM_info, RB_variables, "MA")

  RB_variables.steady_info.offline_time += assembly_time
  save_affine_structures(ROM_info, RB_variables)
  save_M_DEIM_structures(ROM_info, RB_variables)

end

function save_affine_structures(ROM_info::Problem, RB_variables::PoissonSTPGRB)

  if ROM_info.save_offline_structures
    save_CSV(Mₙ, joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    save_CSV([RB_variables.Qᵐ], joinpath(ROM_info.paths.ROM_structures_path, "Qᵐ.csv"))
    save_CSV(MAₙ, joinpath(ROM_info.paths.ROM_structures_path, "MAₙ.csv"))
  end

end

function get_affine_structures(ROM_info::Problem, RB_variables::PoissonSTPGRB)

  operators = []

  push!(operators, get_Mₙ(ROM_info, RB_variables))
  push!(operators, get_MAₙ(ROM_info, RB_variables))
  push!(operators, get_affine_structures(ROM_info, RB_variables.steady_info))

  operators

end

function save_affine_structures(ROM_info::Problem, RB_variables::PoissonSTPGRB)

  if ROM_info.save_offline_structures
    save_CSV(Mₙ, joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    save_CSV([RB_variables.Qᵐ], joinpath(ROM_info.paths.ROM_structures_path, "Qᵐ.csv"))
  end

end

function get_RB_LHS_blocks(ROM_info::Problem, RB_variables::PoissonSTPGRB, Mₙ, Aₙ, MAₙ)

  @info "Assembling LHS using Crank-Nicolson time scheme"

  θ = ROM_info.θ
  δtθ = ROM_info.δt*θ

  Mat_Mat = Mₙ+(δtθ)^2*Aₙ+(δtθ)*MAₙ+(δtθ)*MAₙ'
  M_Mat = Mₙ+(δtθ)*MAₙ
  M_Mat_Mat_M = M_Mat+M_Mat'

  Φₜᵘ₁ = RB_variables.Φₜᵘ[1:end-1, :]' * RB_variables.Φₜᵘ[1:end-1, :]
  Φₜᵘ₁₂ = RB_variables.Φₜᵘ[1:end-1, :]'*RB_variables.Φₜᵘ[2:end, :]

  block1 = zeros(RB_variables.nᵘ, RB_variables.nᵘ)
  for i_s = 1:RB_variables.steady_info.nₛᵘ
    for i_t = 1:RB_variables.nₜᵘ

      i_st = index_mapping(i_s, i_t, RB_variables)

      for j_s = 1:RB_variables.steady_info.nₛᵘ
        for j_t = 1:RB_variables.nₜᵘ
          j_st = index_mapping(j_s, j_t, RB_variables)
          block1[i_st, j_st] += θ^2*Mat_Mat[i_s,j_s]*(i_t===j_t) + (1-θ)^2*Mat_Mat[i_s,j_s]*Φₜᵘ₁[i_t,j_t] - (1-θ)*M_Mat_Mat_M[i_s,j_s]*Φₜᵘ₁[i_t,j_t] + θ*(1-θ)*M_Mat_Mat_M[i_s,j_s]*Φₜᵘ₁₂[i_t,j_t] + θ*(1-θ)*Mat_Mat[i_s,j_s]*Φₜᵘ₁₂[j_t,i_t] + θ*Mat_Mat[j_s,i_s]*Φₜᵘ₁₂[j_t,i_t] + Mₙ*Φₜᵘ₁[i_t,j_t]
        end
      end

    end
  end

  push!(RB_variables.steady_info.LHSₙ, block1)

end

#= function get_RB_RHS_blocks(ROM_info, RB_variables::PoissonSTPGRB, param)

  @info "Assembling RHS"

  Ffun = assemble_forcing(FE_space, param)
  F_mat = zeros(RB_variables.steady_info.Nₛᵘ, RB_variables.Nₜ + 1)
  for (i, tᵢ) in enumerate(ROM_info.t₀:ROM_info.δt:ROM_info.T)
    F_mat[:, i] = Ffun(tᵢ)
  end
  F = (F_mat[:, 2:end] + F_mat[:, 1:end-1])*ROM_info.δt/2
  Fₙ = (RB_variables.steady_info.Φₛᵘ)' * (F * RB_variables.Φₜᵘ)
  push!(RB_variables.steady_info.RHSₙ, reshape(Fₙ', :, 1))

end =#

function get_RB_RHS_blocks(ROM_info::Problem, RB_variables::PoissonSTPGRB, Fₙ, Hₙ)

  @info "Assembling RHS"

  δtθ = ROM_info.δt*ROM_info.θ
  FHₙ = δtθ*(Fₙ+Hₙ)
  push!(RB_variables.steady_info.RHSₙ, reshape(FHₙ', :, 1))

end

function get_RB_system(ROM_info::Problem, RB_variables::PoissonSTPGRB, param)

  @info "Preparing the RB system: fetching reduced LHS"
  initialize_RB_system(RB_variables)
  θᵐ, θᵐᵃ, θᵃ, θᶠ = get_θ(ROM_info, RB_variables, param)
  blocks = [1]
  operators = get_system_blocks(ROM_info, RB_variables, blocks, blocks)

  if "LHS" ∈ operators
    Mₙ = assemble_online_structure(θᵐ, RB_variables.Mₙ)
    Aₙ = assemble_online_structure(θᵃ, RB_variables.steady_info.Aₙ)
    MAₙ = assemble_online_structure(θᵐᵃ, RB_variables.MAₙ)
    get_RB_LHS_blocks(ROM_info, RB_variables, Mₙ, Aₙ, MAₙ)
  end

  if "RHS" ∈ operators
    if !ROM_info.build_parametric_RHS
      @info "Preparing the RB system: fetching reduced RHS"
      Fₙ_μ = assemble_online_structure(θᶠ, RB_variables.steady_info.Fₙ)
      Hₙ_μ = assemble_online_structure(θʰ, RB_variables.steady_info.Hₙ)
      push!(RB_variables.steady_info.RHSₙ, reshape(Fₙ_μ+Hₙ_μ,:,1))
    else
      @info "Preparing the RB system: assembling reduced RHS exactly"
      Fₙ_μ, Hₙ_μ = build_param_RHS(ROM_info, RB_variables, param)
      push!(RB_variables.steady_info.RHSₙ, reshape(Fₙ_μ+Hₙ_μ,:,1))
    end
  end

end

function build_param_RHS(ROM_info::Problem, RB_variables::PoissonSTPGRB, param)

  δtθ = ROM_info.δt*ROM_info.θ

  FE_space = get_FE_space(problem_info, param.model)
  F_t, H_t = assemble_forcing(FE_space, RB_variables, param)
  F, H = zeros(RB_variables.steady_info.Nₛᵘ, RB_variables.Nₜ), zeros(RB_variables.steady_info.Nₛᵘ, RB_variables.Nₜ)
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ
  for (i, tᵢ) in enumerate(times_θ)
    F[:,i] = F_t(tᵢ)
    H[:,i] = H_t(tᵢ)
  end
  F *= δtθ
  H *= δtθ

  θᵃ_temp = get_θᵃ(ROM_info, RB_variables.steady_info, param)
  AΦᵀPᵤ⁻¹ = assemble_online_structure(θᵃ_temp, RB_variables.steady_info.AΦᵀPᵤ⁻¹)

  Fₙ = AΦᵀPᵤ⁻¹*(F*RB_variables.Φₜᵘ)
  Hₙ = AΦᵀPᵤ⁻¹*(H*RB_variables.Φₜᵘ)

  reshape(Fₙ, :, 1), reshape(Hₙ, :, 1)

end

function get_θ(ROM_info::Problem, RB_variables::PoissonSTPGRB, param)

  θᵐ_temp = get_θᵐ(ROM_info, RB_variables, param)
  θᵃ_temp = get_θᵃ(ROM_info, RB_variables, param)

  Qᵐ, Qᵃ = length(θᵐ_temp), length(θᵃ_temp)
  θᵐ = [θᵐ_temp[q₁]*θᵐ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qᵐ]
  θᵐᵃ = [θᵐ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qᵃ]
  θᵃᵐ = [θᵐ_temp[q₁]*θᵃ_temp[q₂] for q₂ = 1:Qᵃ for q₁ = 1:Qᵐ]

  θᵃ_temp = get_θᵃ(ROM_info, RB_variables, param)
  θᵃ = [θᵃ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qᵃ]

  if !ROM_info.build_parametric_RHS

    θᶠ_temp, θʰ_temp = get_θᶠʰ(ROM_info, RB_variables, param)
    Qᶠ, Qʰ = length(θᶠ_temp), length(θʰ_temp)
    θᶠ = [θᵃ_temp[q₁]*θᶠ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qᶠ]
    θʰ = [θᵃ_temp[q₁]*θʰ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qʰ]

  else

    θᶠ, θʰ = Float64[], Float64[]

  end

  return θᵐ, θᵐᵃ, θᵃᵐ, θᵃ, θᶠ, θʰ

end
