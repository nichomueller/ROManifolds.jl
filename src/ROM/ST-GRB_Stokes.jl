
function primal_supremizers(ROM_info::Info, RB_variables::StokesSTGRB)

  @info "Computing primal supremizers"

  dir_idx = abs.(diag(RB_variables.Xᵘ) .- 1) .< 1e-16

  constraint_mat = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "primalConstraint.csv"))
  constraint_mat[dir_idx] = 0

  supr_primal = Matrix(solve(PardisoSolver(), RB_variables.Xᵘ, constraint_mat * RB_variables.Φₛᵖ))

  min_norm = 1e16
  for i = 1:size(supr_primal)[2]

    @info "Normalizing primal supremizer $i"

    for j in 1:RB_variables.P.S.nₛᵘ
      supr_primal[:, i] -= mydot(supr_primal[:, i], RB_variables.P.S.Φₛᵘ[:,j], RB_variables.P.S.Xᵘ₀) / mynorm(RB_variables.P.S.Φₛᵘ[:,j], RB_variables.P.S.Xᵘ₀) * RB_variables.P.S.Φₛᵘ[:,j]
    end
    for j in range(i)
      supr_primal[:, i] -= mydot(supr_primal[:, i], supr_primal[:, j], RB_variables.P.S.Xᵘ₀) / mynorm(supr_primal[:, j], RB_variables.P.S.Xᵘ₀) * supr_primal[:, j]
    end

    supr_norm = mynorm(supr_primal[:, i], RB_variables.P.S.Xᵘ₀)
    min_norm = min(supr_norm, min_norm)
    @info "Norm supremizers: $supr_norm"
    supr_primal[:, i] /= supr_norm

  end

  @info "Primal supremizers enrichment ended with norm: $min_norm"

  supr_primal[abs.(supr_primal) < 1e-15] = 0
  RB_variables.P.S.Φₛᵘ = hcat(RB_variables.P.S.Φₛᵘ, supr_primal)
  RB_variables.P.S.nₛᵘ = size(RB_variables.P.S.Φₛᵘ)[2]

end

function time_supremizers(RB_variables::StokesSTGRB)

  @info "Checking if primal supremizers in time need to be added"

  ΦₜᵘΦₜ = RB_variables.P.Φₛᵘ' * RB_variables.Φₜᵖ
  nₜ = size(Φₜ)[2]
  crit_idx = Int64[]
  crit_norm = Int64[]

  ΦₜᵘΦₜ[:,1] /= norm(ΦₜᵘΦₜ[:,1])
  for i = 2:nₜ

    for j in 1:nₜ
      ΦₜᵘΦₜ[:, i] -= (ΦₜᵘΦₜ[:, i]' * ΦₜᵘΦₜ[:,j]) / norm(ΦₜᵘΦₜ[:,j]) * ΦₜᵘΦₜ[:,j]
    end

    normᵢ = norm(ΦₜᵘΦₜ[:, i])
    ΦₜᵘΦₜ[:, i] /= normᵢ
    if normᵢ ≤ 1e-2
      @info "Time basis vector number $i of field p needs to be added to the velocity's time basis; the corresponding norm is: $(crit_norm[i])"
      Φₜ_crit = RB_variables.Φₜᵖ[:,crit_idx]
      for j in 1:RB_variables.P.nₜᵘ
        Φₜ_crit -= (Φₜ_crit' * RB_variables.P.Φₜᵘ[:,j]) / norm(RB_variables.P.Φₛᵘ[:,j]) * RB_variables.P.Φₜᵘ[:,j]
      end
      Φₜ_crit /= norm(Φₜ_crit)
      RB_variables.P.Φₜᵘ = hcat(RB_variables.P.Φₜᵘ, Φₜ_crit)
      RB_variables.P.nₜᵘ += 1
    end

  end

end

function perform_supremizer_enrichment_space(ROM_info::Info, RB_variables::StokesSTGRB)

  primal_supremizers(ROM_info, RB_variables)

end

function perform_supremizer_enrichment_time(RB_variables::StokesSTGRB)

  time_supremizers(RB_variables)

end

function build_reduced_basis(ROM_info::Info, RB_variables::StokesSTGRB)

  @info "Building the space-time reduced basis for fields (u,p,λ), using a tolerance of ($(ROM_info.ϵₛ),$(ROM_info.ϵₜ))"

  RB_building_time = @elapsed begin
    PODs_space(ROM_info, RB_variables)
    perform_supremizer_enrichment_space(ROM_info, RB_variables)
    PODs_time(ROM_info, RB_variables)
    perform_supremizer_enrichment_time(RB_variables)
  end
  RB_variables.P.nᵘ = RB_variables.P.S.nₛᵘ * RB_variables.P.nₜᵘ
  RB_variables.P.Nᵘ = RB_variables.P.S.Nₛᵘ * RB_variables.P.Nₜ
  RB_variables.nᵖ = RB_variables.nₛᵖ * RB_variables.nₜᵖ
  RB_variables.Nᵖ = RB_variables.Nₛᵖ * RB_variables.P.Nₜ

  RB_variables.P.S.offline_time += RB_building_time

  if ROM_info.save_offline_structures
    save_CSV(RB_variables.P.S.Φₛᵘ, joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
    save_CSV(RB_variables.P.Φₜᵘ, joinpath(ROM_info.paths.basis_path, "Φₜᵘ.csv"))
    save_CSV(RB_variables.Φₛᵖ, joinpath(ROM_info.paths.basis_path, "Φₛᵖ.csv"))
    save_CSV(RB_variables.Φₜᵖ, joinpath(ROM_info.paths.basis_path, "Φₜᵖ.csv"))
  end

end

function get_Aₙ(ROM_info::Info, RB_variables::StokesSTGRB) :: Vector

  return get_Aₙ(ROM_info, RB_variables.P)

end

function get_Mₙ(ROM_info::Info, RB_variables::StokesSTGRB) :: Vector

  return get_Mₙ(ROM_info, RB_variables.P)

end

function get_Bₙ(ROM_info::Info, RB_variables::StokesSTGRB) :: Vector

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Bₙ.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Bᵀₙ.csv"))
    @info "Importing reduced affine velocity-pressure and pressure-velocity matrices"
    Bₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Bₙ.csv"))
    RB_variables.Bₙ = reshape(Bₙ,RB_variables.S.nₛᵘ,RB_variables.nₛᵖ,1)
    Bᵀₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Bᵀₙ.csv"))
    RB_variables.Bᵀₙ = reshape(Bᵀₙ,RB_variables.nₛᵖ,RB_variables.S.nₛᵘ,1)
    return []
  else
    @info "Failed to import the reduced affine velocity-pressure and pressure-velocity matrices: must build them"
    return ["B"]
  end

end

function assemble_affine_matrices(ROM_info::Info, RB_variables::StokesSTGRB, var::String)

  if var === "B"
    @info "Assembling affine reduced velocity-pressure and pressure-velocity matrices"
    B = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "B.csv"); convert_to_sparse = true)
    Bₙ = (RB_variables.Φₛᵖ)' * B * RB_variables.S.Φₛᵘ
    RB_variables.Bₙ = zeros(RB_variables.nₛᵖ, RB_variables.S.nₛᵘ, 1)
    RB_variables.Bₙ[:,:,1] = Bₙ
    Bᵀ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Bᵀ.csv"); convert_to_sparse = true)
    Bᵀₙ = (RB_variables.S.Φₛᵘ)' * Bᵀ * RB_variables.Φₛᵖ
    RB_variables.Bᵀₙ = zeros(RB_variables.S.nₛᵘ, RB_variables.nₛᵖ, 1)
    RB_variables.Bᵀₙ[:,:,1] = Bᵀₙ
  else
    assemble_affine_matrices(ROM_info, RB_variables.P, var)
  end

end

function assemble_MDEIM_matrices(ROM_info::Info, RB_variables::StokesSTGRB, var::String)

  assemble_MDEIM_matrices(ROM_info, RB_variables.P, var)

end

function assemble_affine_vectors(ROM_info::Info, RB_variables::StokesSTGRB, var::String)

  assemble_affine_vectors(ROM_info, RB_variables.P, var)

end

function assemble_DEIM_vectors(ROM_info::Info, RB_variables::StokesSTGRB, var::String)

  assemble_DEIM_vectors(ROM_info, RB_variables.P, var)

end

function assemble_offline_structures(ROM_info::Info, RB_variables::StokesSTGRB, operators=nothing)

  if isnothing(operators)
    operators = set_operators(ROM_info, RB_variables)
  end

  assembly_time = 0
  if "B" ∈ operators
    assembly_time += @elapsed begin
      assemble_affine_matrices(ROM_info, RB_variables, "B")
    end
  end

  RB_variables.S.offline_time += assembly_time

  assemble_offline_structures(ROM_info, RB_variables.P, operators)
  save_affine_structures(ROM_info, RB_variables)
  save_M_DEIM_structures(ROM_info, RB_variables)

end

function save_affine_structures(ROM_info::Info, RB_variables::StokesSTGRB)

  if ROM_info.save_offline_structures
    save_CSV(RB_variables.Bₙ, joinpath(ROM_info.paths.ROM_structures_path, "Bₙ.csv"))
    save_CSV(RB_variables.Bᵀₙ, joinpath(ROM_info.paths.ROM_structures_path, "Bᵀₙ.csv"))
    save_affine_structures(ROM_info, RB_variables.P)
  end

end

function get_affine_structures(ROM_info::Info, RB_variables::StokesSTGRB) :: Vector

  operators = String[]
  append!(operators, get_Bₙ(ROM_info, RB_variables))
  append!(operators, get_affine_structures(ROM_info, RB_variables.P))

  return operators

end

function get_RB_LHS_blocks(ROM_info, RB_variables::StokesSTGRB, θᵐ, θᵃ)

  @info "Assembling LHS using Crank-Nicolson time scheme"

  θ = ROM_info.θ
  δtθ = ROM_info.δt*θ
  nₜᵘ = RB_variables.P.nₜᵘ
  Qᵐ = RB_variables.P.Qᵐ
  Qᵃ = RB_variables.P.S.Qᵃ

  Φₜᵘ_M = zeros(RB_variables.P.nₜᵘ, RB_variables.P.nₜᵘ, Qᵐ)
  Φₜᵘ₁_M = zeros(RB_variables.P.nₜᵘ, RB_variables.P.nₜᵘ, Qᵐ)
  Φₜᵘ_A = zeros(RB_variables.P.nₜᵘ, RB_variables.P.nₜᵘ, Qᵃ)
  Φₜᵘ₁_A = zeros(RB_variables.P.nₜᵘ, RB_variables.P.nₜᵘ, Qᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = sum(RB_variables.P.Φₜᵘ[:,i_t].*RB_variables.P.Φₜᵘ[:,j_t].*θᵐ[q,:]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_M[i_t,j_t,q] = sum(RB_variables.P.Φₜᵘ[2:end,i_t].*RB_variables.P.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = sum(RB_variables.P.Φₜᵘ[:,i_t].*RB_variables.P.Φₜᵘ[:,j_t].*θᵃ[q,:]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_A[i_t,j_t,q] = sum(RB_variables.P.Φₜᵘ[2:end,i_t].*RB_variables.P.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  ΦₜᵘΦₜᵖ = RB_variables.P.Φₜᵘ' * RB_variables.Φₜᵖ

  block₁ = zeros(RB_variables.nᵘ, RB_variables.nᵘ)
  block₂ = zeros(RB_variables.nᵘ, RB_variables.nᵖ)
  block₃ = zeros(RB_variables.nᵖ, RB_variables.nᵘ)

  for i_s = 1:RB_variables.P.S.nₛᵘ
    for i_t = 1:RB_variables.P.nₜᵘ

      i_st = index_mapping(i_s, i_t, RB_variables)

      for j_s = 1:RB_variables.P.S.nₛᵘ
        for j_t = 1:RB_variables.P.nₜᵘ

          j_st = index_mapping(j_s, j_t, RB_variables)

          Aₙ_μ_i_j = δtθ*RB_variables.S.Aₙ[i_s,j_s,:]'*Φₜᵘ_A[i_t,j_t,:]
          Mₙ_μ_i_j = RB_variables.Mₙ[i_s,j_s,:]'*Φₜᵘ_M[i_t,j_t,:]
          Aₙ₁_μ_i_j = δtθ*RB_variables.S.Aₙ[i_s,j_s,:]'*Φₜᵘ₁_A[i_t,j_t,:]
          Mₙ₁_μ_i_j = RB_variables.Mₙ[i_s,j_s,:]'*Φₜᵘ₁_M[i_t,j_t,:]

          block₁[i_st,j_st] = θ*(Aₙ_μ_i_j+Mₙ_μ_i_j) + (1-θ)*Aₙ₁_μ_i_j - θ*Mₙ₁_μ_i_j

        end
      end

      for j_s = 1:RB_variables.nₛᵖ
        for j_t = 1:RB_variables.nₜᵖ
          j_st = index_mapping(j_s, j_t, RB_variables, "p")
          block₂[i_st,j_st] = δtθ*RB_variables.Bᵀₙ[i_s,j_s]*ΦₜᵘΦₜᵖ[i_t,j_t]
        end
      end

    end
  end

  for i_s = 1:RB_variables.nₛᵖ
    for i_t = 1:RB_variables.nₜᵖ

      i_st = index_mapping(i_s, i_t, RB_variables, "p")

      for j_s = 1:RB_variables.P.S.nₛᵘ
        for j_t = 1:RB_variables.P.nₜᵘ
          j_st = index_mapping(j_s, j_t, RB_variables)
          block₃[i_st,j_st] = RB_variables.Bₙ[i_s,j_s]*ΦₜᵘΦₜᵖ[j_t,i_t]
        end
      end

    end
  end

  push!(RB_variables.P.S.LHSₙ, block₁)
  push!(RB_variables.P.S.LHSₙ, block₂)
  push!(RB_variables.P.S.LHSₙ, block₃)
  push!(RB_variables.P.S.LHSₙ, Matrix{Float64}[])

end

function get_RB_RHS_blocks(ROM_info::Info, RB_variables::StokesSTGRB, θᶠ::Array, θʰ::Array)

  @info "Assembling RHS"

  get_RB_RHS_blocks(ROM_info, RB_variables.P, θᶠ, θʰ)

  block₂ = zeros(RB_variables.nᵖ,1)
  push!(RB_variables.P.S.RHSₙ, block₂)

end

function get_RB_system(ROM_info::Info, RB_variables::StokesSTGRB, param)

  @info "Preparing the RB system: fetching reduced LHS"
  initialize_RB_system(RB_variables.S)
  get_Q(ROM_info, RB_variables)
  blocks = [1]
  operators = get_system_blocks(ROM_info, RB_variables, blocks, blocks)

  if ROM_info.space_time_M_DEIM
    θᵐ, θᵃ, θᶠ, θʰ = get_θₛₜ(ROM_info, RB_variables, param)
  else
    θᵐ, θᵃ, θᶠ, θʰ = get_θ(ROM_info, RB_variables, param)
  end

  if "LHS" ∈ operators
    get_RB_LHS_blocks(ROM_info, RB_variables, θᵐ, θᵃ)
  end

  if "RHS" ∈ operators
    if !ROM_info.build_parametric_RHS
      @info "Preparing the RB system: fetching reduced RHS"
      get_RB_RHS_blocks(ROM_info, RB_variables, θᶠ, θʰ)
    else
      @info "Preparing the RB system: assembling reduced RHS exactly"
      build_param_RHS(ROM_info, RB_variables, param)
    end
  end

end

function build_param_RHS(ROM_info::Info, RB_variables::StokesSTGRB, param)

  build_param_RHS(ROM_info, RB_variables.P, param)
  push!(RB_variables.S.RHSₙ, zeros(RB_variables.nᵖ,1))

end

function get_θ(ROM_info::Info, RB_variables::StokesSTGRB, param) :: Tuple

  return get_θ(ROM_info, RB_variables.P, param)

end

function get_θₛₜ(ROM_info::Info, RB_variables::StokesSTGRB, param) :: Tuple

  return get_θₛₜ(ROM_info, RB_variables.P, param)

end
