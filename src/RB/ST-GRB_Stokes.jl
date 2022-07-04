include("RBPoisson_unsteady.jl")

function primal_supremizers(RBInfo::Info, RBVars::StokesSTGRB)

  println("Computing primal supremizers")

  dir_idx = abs.(diag(RBVars.Xᵘ) .- 1) .< 1e-16

  constraint_mat = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.FEM_structures_path, "B.csv"); true)'
  constraint_mat[dir_idx[dir_idx≤RBVars.P.S.Nₛᵘ*RBVars.P.S.Nₛᵖ]] = 0

  supr_primal = Matrix{Float64}(solve(PardisoSolver(), RBVars.Xᵘ, constraint_mat * RBVars.Φₛᵖ))

  min_norm = 1e16
  for i = 1:size(supr_primal)[2]

    println("Normalizing primal supremizer $i")

    for j in 1:RBVars.P.S.nₛᵘ
      supr_primal[:, i] -= mydot(supr_primal[:, i], RBVars.P.S.Φₛᵘ[:,j], RBVars.P.S.Xᵘ₀) /
      mynorm(RBVars.P.S.Φₛᵘ[:,j], RBVars.P.S.Xᵘ₀) * RBVars.P.S.Φₛᵘ[:,j]
    end
    for j in range(i)
      supr_primal[:, i] -= mydot(supr_primal[:, i], supr_primal[:, j], RBVars.P.S.Xᵘ₀) /
      mynorm(supr_primal[:, j], RBVars.P.S.Xᵘ₀) * supr_primal[:, j]
    end

    supr_norm = mynorm(supr_primal[:, i], RBVars.P.S.Xᵘ₀)
    min_norm = min(supr_norm, min_norm)
    println("Norm supremizers: $supr_norm")
    supr_primal[:, i] /= supr_norm

  end

  println("Primal supremizers enrichment ended with norm: $min_norm")

  supr_primal[abs.(supr_primal) < 1e-15] = 0
  RBVars.P.S.Φₛᵘ = hcat(RBVars.P.S.Φₛᵘ, supr_primal)
  RBVars.P.S.nₛᵘ = size(RBVars.P.S.Φₛᵘ)[2]

end

function time_supremizers(RBVars::StokesSTGRB)

  println("Checking if primal supremizers in time need to be added")

  ΦₜᵘΦₜ = RBVars.P.Φₛᵘ' * RBVars.Φₜᵖ
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
      println("Time basis vector number $i of field p needs to be added to the velocity's time basis; the corresponding norm is: $(crit_norm[i])")
      Φₜ_crit = RBVars.Φₜᵖ[:,crit_idx]
      for j in 1:RBVars.P.nₜᵘ
        Φₜ_crit -= (Φₜ_crit' * RBVars.P.Φₜᵘ[:,j]) / norm(RBVars.P.Φₛᵘ[:,j]) * RBVars.P.Φₜᵘ[:,j]
      end
      Φₜ_crit /= norm(Φₜ_crit)
      RBVars.P.Φₜᵘ = hcat(RBVars.P.Φₜᵘ, Φₜ_crit)
      RBVars.P.nₜᵘ += 1
    end

  end

end

function perform_supremizer_enrichment_space(RBInfo::Info, RBVars::StokesSTGRB)

  primal_supremizers(RBInfo, RBVars)

end

function perform_supremizer_enrichment_time(RBVars::StokesSTGRB)

  time_supremizers(RBVars)

end

function build_reduced_basis(RBInfo::Info, RBVars::StokesSTGRB)

  println("Building the space-time reduced basis for fields (u,p,λ), using a tolerance of ($(RBInfo.ϵₛ),$(RBInfo.ϵₜ))")

  RB_building_time = @elapsed begin
    PODs_space(RBInfo, RBVars)
    perform_supremizer_enrichment_space(RBInfo, RBVars)
    PODs_time(RBInfo, RBVars)
    perform_supremizer_enrichment_time(RBVars)
  end
  RBVars.P.nᵘ = RBVars.P.S.nₛᵘ * RBVars.P.nₜᵘ
  RBVars.P.Nᵘ = RBVars.P.S.Nₛᵘ * RBVars.P.Nₜ
  RBVars.nᵖ = RBVars.nₛᵖ * RBVars.nₜᵖ
  RBVars.Nᵖ = RBVars.Nₛᵖ * RBVars.P.Nₜ

  RBVars.P.S.offline_time += RB_building_time

  if RBInfo.save_offline_structures
    save_CSV(RBVars.P.S.Φₛᵘ, joinpath(RBInfo.paths.basis_path, "Φₛᵘ.csv"))
    save_CSV(RBVars.P.Φₜᵘ, joinpath(RBInfo.paths.basis_path, "Φₜᵘ.csv"))
    save_CSV(RBVars.Φₛᵖ, joinpath(RBInfo.paths.basis_path, "Φₛᵖ.csv"))
    save_CSV(RBVars.Φₜᵖ, joinpath(RBInfo.paths.basis_path, "Φₜᵖ.csv"))
  end

end

function get_Aₙ(RBInfo::Info, RBVars::StokesSTGRB) :: Vector

  return get_Aₙ(RBInfo, RBVars.P)

end

function get_Mₙ(RBInfo::Info, RBVars::StokesSTGRB) :: Vector

  return get_Mₙ(RBInfo, RBVars.P)

end

function get_Bₙ(RBInfo::Info, RBVars::StokesSTGRB) :: Vector

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Bₙ.csv")) && isfile(joinpath(RBInfo.paths.ROM_structures_path, "Bᵀₙ.csv"))
    println("Importing reduced affine velocity-pressure and pressure-velocity matrices")
    Bₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.paths.ROM_structures_path, "Bₙ.csv"))
    RBVars.Bₙ = reshape(Bₙ,RBVars.S.nₛᵘ,RBVars.nₛᵖ,1)
    return [""]
  else
    println("Failed to import the reduced affine velocity-pressure and pressure-velocity matrices: must build them")
    return ["B"]
  end

end

function assemble_affine_matrices(RBInfo::Info, RBVars::StokesSTGRB, var::String)

  if var == "B"
    println("Assembling affine reduced velocity-pressure and pressure-velocity matrices")
    B = load_CSV(sparse([],[],T[]), joinpath(RBInfo.paths.FEM_structures_path, "B.csv"))
    Bₙ = (RBVars.Φₛᵖ)' * B * RBVars.S.Φₛᵘ
    RBVars.Bₙ = zeros(RBVars.nₛᵖ, RBVars.S.nₛᵘ, 1)
    RBVars.Bₙ[:,:,1] = Bₙ
  else
    assemble_affine_matrices(RBInfo, RBVars.P, var)
  end

end

function assemble_MDEIM_matrices(RBInfo::Info, RBVars::StokesSTGRB, var::String)

  assemble_MDEIM_matrices(RBInfo, RBVars.P, var)

end

function assemble_affine_vectors(RBInfo::Info, RBVars::StokesSTGRB, var::String)

  assemble_affine_vectors(RBInfo, RBVars.P, var)

end

function assemble_DEIM_vectors(RBInfo::Info, RBVars::StokesSTGRB, var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.P, var)

end

function assemble_offline_structures(RBInfo::Info, RBVars::StokesSTGRB, operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assembly_time = 0
  if "B" ∈ operators
    assembly_time += @elapsed begin
      assemble_affine_matrices(RBInfo, RBVars, "B")
    end
  end

  RBVars.S.offline_time += assembly_time

  assemble_offline_structures(RBInfo, RBVars.P, operators)
  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function save_affine_structures(RBInfo::Info, RBVars::StokesSTGRB)

  if RBInfo.save_offline_structures
    save_CSV(RBVars.Bₙ, joinpath(RBInfo.paths.ROM_structures_path, "Bₙ.csv"))
    save_CSV(RBVars.Bᵀₙ, joinpath(RBInfo.paths.ROM_structures_path, "Bᵀₙ.csv"))
    save_affine_structures(RBInfo, RBVars.P)
  end

end

function get_affine_structures(RBInfo::Info, RBVars::StokesSTGRB) :: Vector

  operators = String[]
  append!(operators, get_Bₙ(RBInfo, RBVars))
  append!(operators, get_affine_structures(RBInfo, RBVars.P))

  return operators

end

function get_RB_LHS_blocks(RBInfo, RBVars::StokesSTGRB, θᵐ, θᵃ)

  println("Assembling LHS using Crank-Nicolson time scheme")

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ
  nₜᵘ = RBVars.P.nₜᵘ
  Qᵐ = RBVars.P.Qᵐ
  Qᵃ = RBVars.P.S.Qᵃ

  Φₜᵘ_M = zeros(RBVars.P.nₜᵘ, RBVars.P.nₜᵘ, Qᵐ)
  Φₜᵘ₁_M = zeros(RBVars.P.nₜᵘ, RBVars.P.nₜᵘ, Qᵐ)
  Φₜᵘ_A = zeros(RBVars.P.nₜᵘ, RBVars.P.nₜᵘ, Qᵃ)
  Φₜᵘ₁_A = zeros(RBVars.P.nₜᵘ, RBVars.P.nₜᵘ, Qᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = sum(RBVars.P.Φₜᵘ[:,i_t].*RBVars.P.Φₜᵘ[:,j_t].*θᵐ[q,:]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_M[i_t,j_t,q] = sum(RBVars.P.Φₜᵘ[2:end,i_t].*RBVars.P.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = sum(RBVars.P.Φₜᵘ[:,i_t].*RBVars.P.Φₜᵘ[:,j_t].*θᵃ[q,:]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_A[i_t,j_t,q] = sum(RBVars.P.Φₜᵘ[2:end,i_t].*RBVars.P.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  ΦₜᵘΦₜᵖ = RBVars.P.Φₜᵘ' * RBVars.Φₜᵖ

  block₁ = zeros(RBVars.nᵘ, RBVars.nᵘ)
  block₂ = zeros(RBVars.nᵘ, RBVars.nᵖ)
  block₃ = zeros(RBVars.nᵖ, RBVars.nᵘ)

  for i_s = 1:RBVars.P.S.nₛᵘ
    for i_t = 1:RBVars.P.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      for j_s = 1:RBVars.P.S.nₛᵘ
        for j_t = 1:RBVars.P.nₜᵘ

          j_st = index_mapping(j_s, j_t, RBVars)

          Aₙ_μ_i_j = δtθ*RBVars.S.Aₙ[i_s,j_s,:]'*Φₜᵘ_A[i_t,j_t,:]
          Mₙ_μ_i_j = RBVars.Mₙ[i_s,j_s,:]'*Φₜᵘ_M[i_t,j_t,:]
          Aₙ₁_μ_i_j = δtθ*RBVars.S.Aₙ[i_s,j_s,:]'*Φₜᵘ₁_A[i_t,j_t,:]
          Mₙ₁_μ_i_j = RBVars.Mₙ[i_s,j_s,:]'*Φₜᵘ₁_M[i_t,j_t,:]

          block₁[i_st,j_st] = θ*(Aₙ_μ_i_j+Mₙ_μ_i_j) + (1-θ)*Aₙ₁_μ_i_j - θ*Mₙ₁_μ_i_j

        end
      end

      for j_s = 1:RBVars.nₛᵖ
        for j_t = 1:RBVars.nₜᵖ
          j_st = index_mapping(j_s, j_t, RBVars, "p")
          block₂[i_st,j_st] = δtθ*RBVars.Bᵀₙ[i_s,j_s]*ΦₜᵘΦₜᵖ[i_t,j_t]
        end
      end

    end
  end

  for i_s = 1:RBVars.nₛᵖ
    for i_t = 1:RBVars.nₜᵖ

      i_st = index_mapping(i_s, i_t, RBVars, "p")

      for j_s = 1:RBVars.P.S.nₛᵘ
        for j_t = 1:RBVars.P.nₜᵘ
          j_st = index_mapping(j_s, j_t, RBVars)
          block₃[i_st,j_st] = RBVars.Bₙ[i_s,j_s]*ΦₜᵘΦₜᵖ[j_t,i_t]
        end
      end

    end
  end

  push!(RBVars.P.S.LHSₙ, block₁)
  push!(RBVars.P.S.LHSₙ, block₂)
  push!(RBVars.P.S.LHSₙ, block₃)
  push!(RBVars.P.S.LHSₙ, Matrix{Float64}[])

end

function get_RB_RHS_blocks(RBInfo::Info, RBVars::StokesSTGRB, θᶠ::Vector{Float64}, θʰ::Vector{Float64})

  println("Assembling RHS")

  get_RB_RHS_blocks(RBInfo, RBVars.P, θᶠ, θʰ)

  block₂ = zeros(RBVars.nᵖ,1)
  push!(RBVars.P.S.RHSₙ, block₂)

end

function get_RB_system(RBInfo::Info, RBVars::StokesSTGRB, Param)

  println("Preparing the RB system: fetching reduced LHS")
  initialize_RB_system(RBVars.S)
  get_Q(RBInfo, RBVars)
  blocks = [1]
  operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

  if RBInfo.space_time_M_DEIM
    θᵐ, θᵃ, θᶠ, θʰ = get_θₛₜ(RBInfo, RBVars, Param)
  else
    θᵐ, θᵃ, θᶠ, θʰ = get_θ(RBInfo, RBVars, Param)
  end

  if "LHS" ∈ operators
    get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ)
  end

  if "RHS" ∈ operators
    if !RBInfo.build_parametric_RHS
      println("Preparing the RB system: fetching reduced RHS")
      get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ)
    else
      println("Preparing the RB system: assembling reduced RHS exactly")
      build_param_RHS(RBInfo, RBVars, Param)
    end
  end

end

function build_param_RHS(RBInfo::Info, RBVars::StokesSTGRB, Param)

  build_param_RHS(RBInfo, RBVars.P, Param)
  push!(RBVars.S.RHSₙ, zeros(RBVars.nᵖ,1))

end

function get_θ(RBInfo::Info, RBVars::StokesSTGRB, Param) ::Tuple

  return get_θ(RBInfo, RBVars.P, Param)

end

function get_θₛₜ(RBInfo::Info, RBVars::StokesSTGRB, Param) ::Tuple

  return get_θₛₜ(RBInfo, RBVars.P, Param)

end
