function get_Aₙ(RBInfo::Info, RBVars::PoissonSTGRB) :: Vector

  get_Aₙ(RBInfo, RBVars.S)

end

function get_Mₙ(RBInfo::Info, RBVars::PoissonSTGRB) :: Vector

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    @info "Importing reduced affine mass matrix"
    Mₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    RBVars.Mₙ = reshape(Mₙ,RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,:)
    RBVars.Qᵐ = size(RBVars.Mₙ)[3]
    return []
  else
    @info "Failed to import the reduced affine mass matrix: must build it"
    return ["M"]
  end

end

function assemble_affine_matrices(RBInfo::Info, RBVars::PoissonSTGRB, var::String)

  if var == "M"
    RBVars.Qᵐ = 1
    @info "Assembling affine reduced mass"
    M = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "M.csv");
      convert_to_sparse = true)
    RBVars.Mₙ = zeros(RBVars.S.nₛᵘ, RBVars.S.nₛᵘ, 1)
    RBVars.Mₙ[:,:,1] = (RBVars.S.Φₛᵘ)' * M * RBVars.S.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.S, var)
  end

end

function assemble_MDEIM_matrices(RBInfo::Info, RBVars::PoissonSTGRB, var::String)

  @info "The matrix $var is non-affine:
    running the MDEIM offline phase on $nₛ_MDEIM snapshots"

  MDEIM_mat, MDEIM_idx, MDEIMᵢ_mat, row_idx, sparse_el =
    MDEIM_offline(FEMSpace, RBInfo, var)
  Matₙ, Q = assemble_reduced_mat_MDEIM(MDEIM_mat,row_idx,RBInfo,RBVars,var)

  if var == "M"
    RBVars.Mₙ = Matₙ
    RBVars.MDEIMᵢ_M = MDEIMᵢ_mat
    RBVars.MDEIM_idx_M = MDEIM_idx
    RBVars.sparse_el_M = sparse_el
    RBVars.row_idx_M = row_idx
    RBVars.Qᵐ = Q
  elseif var == "A"
    RBVars.S.Aₙ = Matₙ
    RBVars.S.MDEIMᵢ_A = MDEIMᵢ_mat
    RBVars.S.MDEIM_idx_A = MDEIM_idx
    RBVars.S.sparse_el_A = sparse_el
    RBVars.row_idx_A = row_idx
    RBVars.S.Qᵃ = Q
  else
    error("Unrecognized variable to assemble with MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  MDEIM_mat::Matrix,
  row_idx::Vector,
  RBInfo::Info,
  RBVars::PoissonSTGRB,
  var::String) ::Tuple

  if RBInfo.space_time_M_DEIM
    Nₜ = RBVars.Nₜ
    MDEIM_mat_new = reshape(MDEIM_mat,length(row_idx),RBVars.Nₜ,:)
    Q = size(MDEIM_mat_new)[3]
    r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.S.Nₛᵘ)
    MatqΦ = zeros(RBVars.S.Nₛᵘ,RBVars.S.nₛᵘ,Q*Nₜ)
    for q = 1:Q
      @info "ST-GRB: affine component number $q/$Q, matrix $var"
      for j = 1:RBVars.S.Nₛᵘ
        Mat_idx = findall(x -> x == j, r_idx)
        MatqΦ[j,:,(q-1)*Nₜ+1:q*Nₜ] =
          (MDEIM_mat_new[Mat_idx,:,q]' * RBVars.S.Φₛᵘ[c_idx[Mat_idx],:])'
      end
    end
    Matₙ = reshape(RBVars.S.Φₛᵘ' * reshape(MatqΦ,RBVars.S.Nₛᵘ,:),
      RBVars.S.nₛᵘ,:,Q*Nₜ)
  else
    Q = size(MDEIM_mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.S.Nₛᵘ)
    MatqΦ = zeros(RBVars.S.Nₛᵘ,RBVars.S.nₛᵘ,Q)
    for j = 1:RBVars.S.Nₛᵘ
      Mat_idx = findall(x -> x == j, r_idx)
      MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.S.Φₛᵘ[c_idx[Mat_idx],:])'
    end
    Matₙ = reshape(RBVars.S.Φₛᵘ' *
      reshape(MatqΦ,RBVars.S.Nₛᵘ,:),RBVars.S.nₛᵘ,:,Q)
  end

  Matₙ, Q

end

function assemble_affine_vectors(RBInfo::Info, RBVars::PoissonSTGRB, var::String)

  assemble_affine_vectors(RBInfo, RBVars.S, var)

end

function assemble_DEIM_vectors(RBInfo::Info, RBVars::PoissonSTGRB, var::String)

  @info "ST-GRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots"

  DEIM_mat, DEIM_idx, DEIMᵢ_mat = DEIM_offline(FEMSpace, RBInfo, var)
  Vecₙ,Q = assemble_reduced_mat_DEIM(DEIM_mat,RBInfo,RBVars)

  if var == "F"
    RBVars.S.DEIMᵢ_mat_F = DEIMᵢ_mat
    RBVars.S.DEIM_idx_F = DEIM_idx
    RBVars.S.Qᶠ = Q
    RBVars.S.Fₙ = Vecₙ
  elseif var == "H"
    RBVars.S.DEIMᵢ_mat_H = DEIMᵢ_mat
    RBVars.S.DEIM_idx_H = DEIM_idx
    RBVars.S.Qʰ = Q
    RBVars.S.Hₙ = Vecₙ
  else
    error("Unrecognized vector to assemble with DEIM")
  end

end

function assemble_reduced_mat_DEIM(
  DEIM_mat::Matrix,
  RBInfo::Info,
  RBVars::PoissonSTGRB) ::Tuple

  if RBInfo.space_time_M_DEIM
    Nₜ = RBVars.Nₜ
    DEIM_mat_new = reshape(DEIM_mat,RBVars.S.Nₛᵘ,:)
    Q = Int(size(MDEIM_mat_new)[2]/Nₜ)
    r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.S.Nₛᵘ)
    MatqΦ = zeros(RBVars.S.Nₛᵘ,RBVars.S.nₛᵘ,Q*Nₜ)
    for q = 1:Q
      @info "ST-GRB: affine component number $q/$Q, matrix $var"
      for j = 1:RBVars.S.Nₛᵘ
        Mat_idx = findall(x -> x == j, r_idx)
        MatqΦ[j,:,(q-1)*Nₜ+1:q*Nₜ] =
          (MDEIM_mat_new[Mat_idx,:,q]' * RBVars.S.Φₛᵘ[c_idx[Mat_idx],:])'
      end
    end
    Matₙ = reshape(RBVars.S.Φₛᵘ' * reshape(MatqΦ,RBVars.S.Nₛᵘ,:),
      RBVars.S.nₛᵘ,:,Q*Nₜ)
  else
    Q = size(DEIM_mat)[2]
    Vecₙ = zeros(RBVars.S.nₛᵘ,1,Q)
    for q = 1:Q
      Vecₙ[:,:,q] = RBVars.S.Φₛᵘ' * Vector(DEIM_mat[:, q])
    end
    Vecₙ = reshape(Vecₙ,:,Q)
  end
  Vecₙ,Q
end

function assemble_offline_structures(RBInfo::Info, RBVars::PoissonSTGRB, operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assembly_time = 0
  if "M" ∈ operators
    assembly_time += @elapsed begin
      if !RBInfo.probl_nl["M"]
        assemble_affine_matrices(RBInfo, RBVars, "M")
      else
        assemble_MDEIM_matrices(RBInfo, RBVars, "M")
      end
    end
  end

  if "A" ∈ operators
    assembly_time += @elapsed begin
      if !RBInfo.probl_nl["A"]
        assemble_affine_matrices(RBInfo, RBVars, "A")
      else
        assemble_MDEIM_matrices(RBInfo, RBVars, "A")
      end
    end
  end

  if "F" ∈ operators
    assembly_time += @elapsed begin
      if !RBInfo.probl_nl["f"]
        assemble_affine_vectors(RBInfo, RBVars, "F")
      else
        assemble_DEIM_vectors(RBInfo, RBVars, "F")
      end
    end
  end

  if "H" ∈ operators
    assembly_time += @elapsed begin
      if !RBInfo.probl_nl["h"]
        assemble_affine_vectors(RBInfo, RBVars, "H")
      else
        assemble_DEIM_vectors(RBInfo, RBVars, "H")
      end
    end
  end
  RBVars.S.offline_time += assembly_time

  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function save_affine_structures(RBInfo::Info, RBVars::PoissonSTGRB)

  if RBInfo.save_offline_structures
    Mₙ = reshape(RBVars.Mₙ, :, RBVars.Qᵐ)
    save_CSV(Mₙ, joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    save_affine_structures(RBInfo, RBVars.S)
  end

end

function get_affine_structures(RBInfo::Info, RBVars::PoissonSTGRB) :: Vector

  operators = String[]
  append!(operators, get_Mₙ(RBInfo, RBVars))
  append!(operators, get_affine_structures(RBInfo, RBVars.S))

  return operators

end

function get_RB_LHS_blocks(RBInfo, RBVars::PoissonSTGRB, θᵐ, θᵃ)

  @info "Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)"

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ
  nₜᵘ = RBVars.nₜᵘ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.S.Qᵃ

  Φₜᵘ_M = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)
  Φₜᵘ₁_M = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)
  Φₜᵘ_A = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)
  Φₜᵘ₁_A = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐ[q,:]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵃ[q,:]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  block₁ = zeros(RBVars.nᵘ, RBVars.nᵘ)

  for i_s = 1:RBVars.S.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      for j_s = 1:RBVars.S.nₛᵘ
        for j_t = 1:RBVars.nₜᵘ

          j_st = index_mapping(j_s, j_t, RBVars)

          Aₙ_μ_i_j = δtθ*RBVars.S.Aₙ[i_s,j_s,:]'*Φₜᵘ_A[i_t,j_t,:]
          Mₙ_μ_i_j = RBVars.Mₙ[i_s,j_s,:]'*Φₜᵘ_M[i_t,j_t,:]
          Aₙ₁_μ_i_j = δtθ*RBVars.S.Aₙ[i_s,j_s,:]'*Φₜᵘ₁_A[i_t,j_t,:]
          Mₙ₁_μ_i_j = RBVars.Mₙ[i_s,j_s,:]'*Φₜᵘ₁_M[i_t,j_t,:]

          block₁[i_st,j_st] = θ*(Aₙ_μ_i_j+Mₙ_μ_i_j) + (1-θ)*Aₙ₁_μ_i_j - θ*Mₙ₁_μ_i_j

        end
      end

    end
  end

  push!(RBVars.S.LHSₙ, block₁)

end

function get_RB_LHS_blocks_spacetime(RBInfo, RBVars::PoissonSTGRB, θᵐ, θᵃ)

  @info "Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)"

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ
  Nₜ = RBVars.Nₜ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.S.Qᵃ
  if Qᵐ == size(RBVars.Mₙ)[3]
    Qᵐ = ceil(Int,Qᵐ/Nₜ)
  end
  if Qᵃ == size(RBVars.S.Aₙ)[3]
    Qᵃ = ceil(Int,Qᵃ/Nₜ)
  end
  Nₜᵐ = Int(size(RBVars.Mₙ)[3]/Qᵐ)
  Nₜᵃ = Int(size(RBVars.S.Aₙ)[3]/Qᵃ)

  Φₜᵘ = RBVars.Φₜᵘ'*RBVars.Φₜᵘ
  Φₜᵘ₁ = RBVars.Φₜᵘ[2:end,:]'*RBVars.Φₜᵘ[1:end-1,:]

  Mₙ = reshape(RBVars.Mₙ,RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,Nₜᵐ,Qᵐ)
  Mₙ = reshape(sum(assemble_online_structure(θᵐ,Mₙ),dims=3),
    RBVars.S.nₛᵘ,RBVars.S.nₛᵘ)
  Aₙ = reshape(RBVars.S.Aₙ,RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,Nₜᵃ,Qᵃ)
  Aₙ = reshape(sum(assemble_online_structure(θᵃ,Aₙ),dims=3),
    RBVars.S.nₛᵘ,RBVars.S.nₛᵘ)

  block₁ = zeros(RBVars.nᵘ, RBVars.nᵘ)

  for i_s = 1:RBVars.S.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      for j_s = 1:RBVars.S.nₛᵘ
        for j_t = 1:RBVars.nₜᵘ

          j_st = index_mapping(j_s, j_t, RBVars)

          Aₙ_μ_i_j = δtθ*Aₙ[i_s,j_s]*Φₜᵘ[i_t,j_t]
          Mₙ_μ_i_j = Mₙ[i_s,j_s]*Φₜᵘ[i_t,j_t]
          Aₙ₁_μ_i_j = δtθ*Aₙ[i_s,j_s]*Φₜᵘ₁[i_t,j_t]
          Mₙ₁_μ_i_j = Mₙ[i_s,j_s]*Φₜᵘ₁[i_t,j_t]

          block₁[i_st,j_st] = θ*(Aₙ_μ_i_j+Mₙ_μ_i_j) + (1-θ)*Aₙ₁_μ_i_j - θ*Mₙ₁_μ_i_j

        end
      end

    end
  end

  push!(RBVars.S.LHSₙ, block₁)

end

function get_RB_RHS_blocks(RBInfo::Info, RBVars::PoissonSTGRB, θᶠ, θʰ)

  @info "Assembling RHS using θ-method time scheme, θ=$(RBInfo.θ)"

  Qᶠ = RBVars.S.Qᶠ
  Qʰ = RBVars.S.Qʰ
  δtθ = RBInfo.δt*RBInfo.θ
  nₜᵘ = RBVars.nₜᵘ

  Φₜᵘ_F = zeros(RBVars.nₜᵘ, Qᶠ)
  Φₜᵘ_H = zeros(RBVars.nₜᵘ, Qʰ)
  [Φₜᵘ_F[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θᶠ[q,:]) for q = 1:Qᶠ for i_t = 1:nₜᵘ]
  [Φₜᵘ_H[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θʰ[q,:]) for q = 1:Qʰ for i_t = 1:nₜᵘ]

  block₁ = zeros(RBVars.nᵘ,1)
  for i_s = 1:RBVars.S.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      Fₙ_μ_i_j = RBVars.S.Fₙ[i_s,:]'*Φₜᵘ_F[i_t,:]
      Hₙ_μ_i_j = RBVars.S.Hₙ[i_s,:]'*Φₜᵘ_H[i_t,:]

      block₁[i_st,1] = Fₙ_μ_i_j+Hₙ_μ_i_j

    end
  end

  block₁ *= δtθ
  push!(RBVars.S.RHSₙ, block₁)

end

function get_RB_system(RBInfo::Info, RBVars::PoissonSTGRB, Param)

  @info "Preparing the RB system: fetching reduced LHS"
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
    if RBInfo.space_time_M_DEIM
      get_RB_LHS_blocks_spacetime(RBInfo, RBVars, θᵐ, θᵃ)
    else
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ)
    end
  end

  if "RHS" ∈ operators
    if !RBInfo.build_Parametric_RHS
      @info "Preparing the RB system: fetching reduced RHS"
      get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ)
    else
      @info "Preparing the RB system: assembling reduced RHS exactly"
      build_Param_RHS(RBInfo, RBVars, Param)
    end
  end

end

function build_Param_RHS(RBInfo::Info, RBVars::PoissonSTGRB, Param)

  δtθ = RBInfo.δt*RBInfo.θ

  F_t = assemble_forcing(FEMSpace, RBInfo, Param)
  H_t = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  F, H = zeros(RBVars.S.Nₛᵘ, RBVars.Nₜ), zeros(RBVars.S.Nₛᵘ, RBVars.Nₜ)
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+δtθ
  for (i,tᵢ) in enumerate(timesθ)
    F[:,i] = F_t(tᵢ)
    H[:,i] = H_t(tᵢ)
  end
  F *= δtθ
  H *= δtθ

  Fₙ = RBVars.S.Φₛᵘ'*(F*RBVars.Φₜᵘ)
  Hₙ = RBVars.S.Φₛᵘ'*(H*RBVars.Φₜᵘ)

  push!(RBVars.S.RHSₙ, reshape(Fₙ'+Hₙ',:,1))

end

function get_θ(RBInfo::Info, RBVars::PoissonSTGRB, Param) ::Tuple

  θᵐ = get_θᵐ(RBInfo, RBVars, Param)
  θᵃ = get_θᵃ(RBInfo, RBVars, Param)
  if !RBInfo.build_Parametric_RHS
    θᶠ, θʰ = get_θᶠʰ(RBInfo, RBVars, Param)
  else
    θᶠ, θʰ = Float64[], Float64[]
  end

  return θᵐ, θᵃ, θᶠ, θʰ

end

function get_θₛₜ(RBInfo::Info, RBVars::PoissonSTGRB, Param) ::Tuple

  θᵐ = get_θᵐₛₜ(RBInfo, RBVars, Param)
  θᵃ = get_θᵃₛₜ(RBInfo, RBVars, Param)
  if !RBInfo.build_Parametric_RHS
    θᶠ, θʰ = get_θᶠʰₛₜ(RBInfo, RBVars, Param)
  else
    θᶠ, θʰ = Float64[], Float64[]
  end

  return θᵐ, θᵃ, θᶠ, θʰ

end
