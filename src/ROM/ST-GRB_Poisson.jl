include("S-GRB_Poisson.jl")

function get_Aₙ(ROM_info::Info, RB_variables::PoissonSTGRB) :: Vector

  get_Aₙ(ROM_info, RB_variables.S)

end

function get_Mₙ(ROM_info::Info, RB_variables::PoissonSTGRB) :: Vector

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    @info "Importing reduced affine mass matrix"
    Mₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    RB_variables.Mₙ = reshape(Mₙ,RB_variables.S.nₛᵘ,RB_variables.S.nₛᵘ,:)
    RB_variables.Qᵐ = size(RB_variables.Mₙ)[3]
    return []
  else
    @info "Failed to import the reduced affine mass matrix: must build it"
    return ["M"]
  end

end

function assemble_affine_matrices(ROM_info::Info, RB_variables::PoissonSTGRB, var::String)

  if var === "M"
    RB_variables.Qᵐ = 1
    @info "Assembling affine reduced mass"
    M = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "M.csv"); convert_to_sparse = true)
    Mₙ = (RB_variables.S.Φₛᵘ)' * M * RB_variables.S.Φₛᵘ
    RB_variables.Mₙ = zeros(RB_variables.S.nₛᵘ, RB_variables.S.nₛᵘ, 1)
    RB_variables.Mₙ[:,:,1] = Mₙ
  else
    assemble_affine_matrices(ROM_info, RB_variables.S, var)
  end

end

function assemble_MDEIM_matrices_standard(ROM_info::Info, RB_variables::PoissonSTGRB, var::String)

  @info "The matrix $var is non-affine: running the MDEIM offline phase on $nₛ_MDEIM snapshots"

  MDEIM_mat, MDEIM_idx, sparse_el, _, _ = MDEIM_offline(FE_space, ROM_info, var)

  Q = size(MDEIM_mat)[2]
  Matₙ = zeros(RB_variables.S.nₛᵘ, RB_variables.S.nₛᵘ, Q)

  for q = 1:Q
    @info "ST-GRB: affine component number $q, matrix $var"
    Matq = reshape(MDEIM_mat[:,q], (RB_variables.S.Nₛᵘ, RB_variables.S.Nₛᵘ))
    Matₙ[:,:,q] = RB_variables.S.Φₛᵘ' * Matrix(Matq) * RB_variables.S.Φₛᵘ
  end
  MDEIMᵢ_mat = Matrix(MDEIM_mat[MDEIM_idx, :])

  if var === "M"
    RB_variables.Mₙ = Matₙ
    RB_variables.MDEIMᵢ_M = MDEIMᵢ_mat
    RB_variables.MDEIM_idx_M = MDEIM_idx
    RB_variables.sparse_el_M = sparse_el
    RB_variables.Qᵐ = Q
  elseif var === "A"
    RB_variables.S.Aₙ = Matₙ
    RB_variables.S.MDEIMᵢ_A = MDEIMᵢ_mat
    RB_variables.S.MDEIM_idx_A = MDEIM_idx
    RB_variables.S.sparse_el_A = sparse_el
    RB_variables.S.Qᵃ = Q
  else
    @error "Unrecognized variable to assemble with MDEIM"
  end

end

function assemble_MDEIM_matrices_spacetime(ROM_info::Info, RB_variables::PoissonSTGRB, var::String)

  @info "The matrix $var is non-affine: running the MDEIM offline phase on $nₛ_MDEIM snapshots"

  MDEIM_mat, MDEIM_idx, MDEIMᵢ_mat, row_idx, sparse_el = MDEIM_offline(FE_space, ROM_info, var)
  Q = size(MDEIM_mat)[2]
  MDEIM_mat_new = reshape(MDEIM_mat,length(row_idx),:)
  Nₜ = Int(size(MDEIM_mat)[1]/Q)

  Matₙ = zeros(RB_variables.S.nₛᵘ, RB_variables.S.nₛᵘ, Q*Nₜ)

  r_idx, c_idx = from_vec_to_mat_idx(row_idx, RB_variables.S.Nₛᵘ)
  for q = 1:Q
    @info "ST-GRB: affine component number $q/$Q at time step $nₜ/$Nₜ, matrix $var"
    MatqΦ = zeros(RB_variables.S.Nₛᵘ,RB_variables.S.nₛᵘ*Nₜ)
    for j = 1:RB_variables.S.Nₛᵘ
      Mat_idx = findall(x -> x == j, r_idx)
      MatqΦ[j,:] = reshape(MDEIM_mat_new[Mat_idx,(q-1)*Nₜ+1:q*Nₜ]' * RB_variables.S.Φₛᵘ[c_idx[Mat_idx],:],1,:)
    end
    Matₙ[:,:,(q-1)*Nₜ+1:q*Nₜ] = reshape(RB_variables.S.Φₛᵘ' * MatqΦ,RB_variables.S.Nₛᵘ,RB_variables.S.nₛᵘ,Nₜ)
  end

  if var === "M"
    RB_variables.Mₙ = Matₙ
    RB_variables.MDEIMᵢ_M = MDEIMᵢ_mat
    RB_variables.MDEIM_idx_M = MDEIM_idx
    RB_variables.sparse_el_M = sparse_el
    RB_variables.row_idx_M = row_idx
    RB_variables.Qᵐ = Q
  elseif var === "A"
    RB_variables.S.Aₙ = Matₙ
    RB_variables.S.MDEIMᵢ_A = MDEIMᵢ_mat
    RB_variables.S.MDEIM_idx_A = MDEIM_idx
    RB_variables.S.sparse_el_A = sparse_el
    RB_variables.row_idx_A = row_idx
    RB_variables.S.Qᵃ = Q
  else
    @error "Unrecognized variable to assemble with MDEIM"
  end

end

function assemble_affine_vectors(ROM_info::Info, RB_variables::PoissonSTGRB, var::String)

  assemble_affine_vectors(ROM_info, RB_variables.S, var)

end

function assemble_DEIM_vectors(ROM_info::Info, RB_variables::PoissonSTGRB, var::String)

  @info "ST-GRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots"

  DEIM_mat, DEIM_idx, _, _ = DEIM_offline(FE_space, ROM_info, var)
  DEIMᵢ_mat = Matrix(DEIM_mat[DEIM_idx, :])
  Q = size(DEIM_mat)[2]
  varₙ = zeros(RB_variables.nₛᵘ,1,Q)
  for q = 1:Q
    varₙ[:,:,q] = RB_variables.Φₛᵘ' * Vector(DEIM_mat[:, q])
  end
  varₙ = reshape(varₙ,:,Q)

  if var === "F"
    RB_variables.DEIMᵢ_mat_F = DEIMᵢ_mat
    RB_variables.DEIM_idx_F = DEIM_idx
    RB_variables.Qᶠ = Q
    RB_variables.Fₙ = varₙ
  elseif var === "H"
    RB_variables.DEIMᵢ_mat_H = DEIMᵢ_mat
    RB_variables.DEIM_idx_H = DEIM_idx
    RB_variables.Qʰ = Q
    RB_variables.Hₙ = varₙ
  else
    @error "Unrecognized vector to assemble with DEIM"
  end

end

function assemble_MDEIM_matrices(ROM_info::Info, RB_variables::PoissonSTGRB, var::String)
  if ROM_info.space_time_M_DEIM
    assemble_MDEIM_matrices_spacetime(ROM_info, RB_variables, var)
  else
    assemble_MDEIM_matrices_standard(ROM_info, RB_variables, var)
  end
end

function assemble_offline_structures(ROM_info::Info, RB_variables::PoissonSTGRB, operators=nothing)

  if isnothing(operators)
    operators = set_operators(ROM_info, RB_variables)
  end

  assembly_time = 0
  if "M" ∈ operators
    assembly_time += @elapsed begin
      if !ROM_info.probl_nl["M"]
        assemble_affine_matrices(ROM_info, RB_variables, "M")
      else
        assemble_MDEIM_matrices(ROM_info, RB_variables, "M")
      end
    end
  end

  if "A" ∈ operators
    assembly_time += @elapsed begin
      if !ROM_info.probl_nl["A"]
        assemble_affine_matrices(ROM_info, RB_variables, "A")
      else
        assemble_MDEIM_matrices(ROM_info, RB_variables, "A")
      end
    end
  end

  if "F" ∈ operators
    assembly_time += @elapsed begin
      if !ROM_info.probl_nl["f"]
        assemble_affine_vectors(ROM_info, RB_variables, "F")
      else
        assemble_DEIM_vectors(ROM_info, RB_variables, "F")
      end
    end
  end

  if "H" ∈ operators
    assembly_time += @elapsed begin
      if !ROM_info.probl_nl["h"]
        assemble_affine_vectors(ROM_info, RB_variables, "H")
      else
        assemble_DEIM_vectors(ROM_info, RB_variables, "H")
      end
    end
  end
  RB_variables.S.offline_time += assembly_time

  save_affine_structures(ROM_info, RB_variables)
  save_M_DEIM_structures(ROM_info, RB_variables)

end

function save_affine_structures(ROM_info::Info, RB_variables::PoissonSTGRB)

  if ROM_info.save_offline_structures
    Mₙ = reshape(RB_variables.Mₙ, :, RB_variables.Qᵐ)
    save_CSV(Mₙ, joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    save_CSV([RB_variables.Qᵐ], joinpath(ROM_info.paths.ROM_structures_path, "Qᵐ.csv"))
    save_affine_structures(ROM_info, RB_variables.S)
  end

end

function get_affine_structures(ROM_info::Info, RB_variables::PoissonSTGRB) :: Vector

  operators = String[]
  append!(operators, get_Mₙ(ROM_info, RB_variables))
  append!(operators, get_affine_structures(ROM_info, RB_variables.S))

  return operators

end

function get_RB_LHS_blocks(ROM_info, RB_variables::PoissonSTGRB, θᵐ, θᵃ)

  @info "Assembling LHS using θ-method time scheme, θ=$(ROM_info.θ)"

  θ = ROM_info.θ
  δtθ = ROM_info.δt*θ
  nₜᵘ = RB_variables.nₜᵘ
  Qᵐ = RB_variables.Qᵐ
  Qᵃ = RB_variables.S.Qᵃ

  Φₜᵘ_M = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐ)
  Φₜᵘ₁_M = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐ)
  Φₜᵘ_A = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵃ)
  Φₜᵘ₁_A = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*RB_variables.Φₜᵘ[:,j_t].*θᵐ[q,:]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_M[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[2:end,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*RB_variables.Φₜᵘ[:,j_t].*θᵃ[q,:]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_A[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[2:end,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  block₁ = zeros(RB_variables.nᵘ, RB_variables.nᵘ)

  for i_s = 1:RB_variables.S.nₛᵘ
    for i_t = 1:RB_variables.nₜᵘ

      i_st = index_mapping(i_s, i_t, RB_variables)

      for j_s = 1:RB_variables.S.nₛᵘ
        for j_t = 1:RB_variables.nₜᵘ

          j_st = index_mapping(j_s, j_t, RB_variables)

          Aₙ_μ_i_j = δtθ*RB_variables.S.Aₙ[i_s,j_s,:]'*Φₜᵘ_A[i_t,j_t,:]
          Mₙ_μ_i_j = RB_variables.Mₙ[i_s,j_s,:]'*Φₜᵘ_M[i_t,j_t,:]
          Aₙ₁_μ_i_j = δtθ*RB_variables.S.Aₙ[i_s,j_s,:]'*Φₜᵘ₁_A[i_t,j_t,:]
          Mₙ₁_μ_i_j = RB_variables.Mₙ[i_s,j_s,:]'*Φₜᵘ₁_M[i_t,j_t,:]

          block₁[i_st,j_st] = θ*(Aₙ_μ_i_j+Mₙ_μ_i_j) + (1-θ)*Aₙ₁_μ_i_j - θ*Mₙ₁_μ_i_j

        end
      end

    end
  end

  push!(RB_variables.S.LHSₙ, block₁)

end

function get_RB_LHS_blocks_spacetime(ROM_info, RB_variables::PoissonSTGRB, θᵐ, θᵃ)

  @info "Assembling LHS using θ-method time scheme, θ=$(ROM_info.θ)"

  θ = ROM_info.θ
  δtθ = ROM_info.δt*θ
  nₜᵘ = RB_variables.nₜᵘ
  Qᵐ = RB_variables.Qᵐ
  Qᵃ = RB_variables.S.Qᵃ
  Nₜᵐ = Int(size(RB_variables.Mₙ)[3]/Qᵐ)
  Nₜᵃ = Int(size(RB_variables.S.Aₙ)[3]/Qᵃ)

  Φₜᵘ_M = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐ)
  Φₜᵘ₁_M = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐ)
  Φₜᵘ_A = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵃ)
  Φₜᵘ₁_A = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = θᵐ[q]*sum(RB_variables.Φₜᵘ[:,i_t].*RB_variables.Φₜᵘ[:,j_t]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_M[i_t,j_t,q] = θᵐ[q]*sum(RB_variables.Φₜᵘ[2:end,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = θᵃ[q]*sum(RB_variables.Φₜᵘ[:,i_t].*RB_variables.Φₜᵘ[:,j_t]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_A[i_t,j_t,q] = θᵃ[q]*sum(RB_variables.Φₜᵘ[2:end,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  block₁ = zeros(RB_variables.nᵘ, RB_variables.nᵘ)

  for i_s = 1:RB_variables.S.nₛᵘ
    for i_t = 1:RB_variables.nₜᵘ

      i_st = index_mapping(i_s, i_t, RB_variables)

      for j_s = 1:RB_variables.S.nₛᵘ
        for j_t = 1:RB_variables.nₜᵘ

          j_st = index_mapping(j_s, j_t, RB_variables)

          for qᵐ = 1:Qᵐ
            Mₙ_μ_i_j += Φₜᵘ_M[i_t,j_t,qᵃ]*sum(RB_variables.Mₙ[i_s,j_s,(qᵐ-1)*Nₜᵐ+1:qᵐ*Nₜᵐ],dims=3)
            Mₙ₁_μ_i_j += Φₜᵘ₁_M[i_t,j_t,qᵃ]*sum(RB_variables.Mₙ[i_s,j_s,(qᵐ-1)*Nₜᵐ+1:qᵐ*Nₜᵐ],dims=3)
          end
          for qᵃ = 1:Qᵃ
            Aₙ_μ_i_j += δtθ*Φₜᵘ_A[i_t,j_t,qᵃ]*sum(RB_variables.S.Aₙ[i_s,j_s,(qᵃ-1)*Nₜᵃ+1:qᵃ*Nₜᵃ],dims=3)
            Aₙ₁_μ_i_j += δtθ*Φₜᵘ₁_A[i_t,j_t,qᵃ]*sum(RB_variables.S.Aₙ[i_s,j_s,(qᵃ-1)*Nₜᵃ+1:qᵃ*Nₜᵃ],dims=3)
          end

          block₁[i_st,j_st] = θ*(Aₙ_μ_i_j+Mₙ_μ_i_j) + (1-θ)*Aₙ₁_μ_i_j - θ*Mₙ₁_μ_i_j

        end
      end

    end
  end

  push!(RB_variables.S.LHSₙ, block₁)

end

function get_RB_RHS_blocks(ROM_info::Info, RB_variables::PoissonSTGRB, θᶠ, θʰ)

  @info "Assembling RHS"

  Qᶠ = RB_variables.S.Qᶠ
  Qʰ = RB_variables.S.Qʰ
  δtθ = ROM_info.δt*ROM_info.θ
  nₜᵘ = RB_variables.nₜᵘ

  Φₜᵘ_F = zeros(RB_variables.nₜᵘ, Qᶠ)
  Φₜᵘ_H = zeros(RB_variables.nₜᵘ, Qʰ)
  [Φₜᵘ_F[i_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*θᶠ[q,:]) for q = 1:Qᶠ for i_t = 1:nₜᵘ]
  [Φₜᵘ_H[i_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*θʰ[q,:]) for q = 1:Qʰ for i_t = 1:nₜᵘ]

  block₁ = zeros(RB_variables.nᵘ,1)
  for i_s = 1:RB_variables.S.nₛᵘ
    for i_t = 1:RB_variables.nₜᵘ

      i_st = index_mapping(i_s, i_t, RB_variables)

      Fₙ_μ_i_j = RB_variables.S.Fₙ[i_s,:]'*Φₜᵘ_F[i_t,:]
      Hₙ_μ_i_j = RB_variables.S.Hₙ[i_s,:]'*Φₜᵘ_H[i_t,:]

      block₁[i_st,1] = Fₙ_μ_i_j+Hₙ_μ_i_j

    end
  end

  block₁ *= δtθ
  push!(RB_variables.S.RHSₙ, block₁)

end

function get_RB_system(ROM_info::Info, RB_variables::PoissonSTGRB, param)

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
    if ROM_info.space_time_M_DEIM
      get_RB_LHS_blocks_spacetime(ROM_info, RB_variables, θᵐ, θᵃ)
    else
      get_RB_LHS_blocks(ROM_info, RB_variables, θᵐ, θᵃ)
    end
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

function build_param_RHS(ROM_info::Info, RB_variables::PoissonSTGRB, param)

  δtθ = ROM_info.δt*ROM_info.θ

  F_t = assemble_forcing(FE_space, ROM_info, param)
  H_t = assemble_neumann_datum(FE_space, ROM_info, param)
  F, H = zeros(RB_variables.S.Nₛᵘ, RB_variables.Nₜ), zeros(RB_variables.S.Nₛᵘ, RB_variables.Nₜ)
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ
  for (i, tᵢ) in enumerate(times_θ)
    F[:,i] = F_t(tᵢ)
    H[:,i] = H_t(tᵢ)
  end
  F *= δtθ
  H *= δtθ

  Fₙ = RB_variables.S.Φₛᵘ'*(F*RB_variables.Φₜᵘ)
  Hₙ = RB_variables.S.Φₛᵘ'*(H*RB_variables.Φₜᵘ)

  push!(RB_variables.S.RHSₙ, Fₙ+Hₙ)

end

function get_θ(ROM_info::Info, RB_variables::PoissonSTGRB, param) :: Tuple

  θᵐ = get_θᵐ(ROM_info, RB_variables, param)
  θᵃ = get_θᵃ(ROM_info, RB_variables, param)
  if !ROM_info.build_parametric_RHS
    θᶠ, θʰ = get_θᶠʰ(ROM_info, RB_variables, param)
  else
    θᶠ, θʰ = Float64[], Float64[]
  end

  return θᵐ, θᵃ, θᶠ, θʰ

end

function get_θₛₜ(ROM_info::Info, RB_variables::PoissonSTGRB, param) :: Tuple

  θᵐ = get_θᵐₛₜ(ROM_info, RB_variables, param)
  θᵃ = get_θᵃₛₜ(ROM_info, RB_variables, param)
  if !ROM_info.build_parametric_RHS
    θᶠ, θʰ = get_θᶠʰₛₜ(ROM_info, RB_variables, param)
  else
    θᶠ, θʰ = Float64[], Float64[]
  end

  return θᵐ, θᵃ, θᶠ, θʰ

end
