function get_Aₙ(RBInfo::Info, RBVars::PoissonSTGRB) :: Vector

  get_Aₙ(RBInfo, RBVars.S)

end

function get_Mₙ(RBInfo::Info, RBVars::PoissonSTGRB) :: Vector

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    @info "Importing reduced affine mass matrix"
    Mₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    RBVars.Mₙ = reshape(Mₙ,RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,:)
    RBVars.Qᵐ = size(RBVars.Mₙ)[end]
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

function assemble_reduced_mat_MDEIM(
  RBInfo::Info,
  RBVars::PoissonSTGRB,
  MDEIM_mat::Matrix,
  row_idx::Vector,
  var::String)

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

  if var == "M"
    RBVars.Mₙ = Matₙ
    RBVars.Qᵐ = Q
  else
    RBVars.S.Aₙ = Matₙ
    RBVars.S.Qᵃ = Q
  end

end

function assemble_affine_vectors(RBInfo::Info, RBVars::PoissonSTGRB, var::String)

  assemble_affine_vectors(RBInfo, RBVars.S, var)

end

function assemble_reduced_mat_DEIM(
  RBInfo::Info,
  RBVars::PoissonSTGRB,
  DEIM_mat::Matrix,
  var::String)

  if RBInfo.space_time_M_DEIM
    Nₜ = RBVars.Nₜ
    DEIM_mat_new = reshape(DEIM_mat,RBVars.S.Nₛᵘ,:)
    Q = Int(size(DEIM_mat_new)[2]/Nₜ)
    Vecₙ = zeros(RBVars.S.nₛᵘ,1,Q*Nₜ)
    for q = 1:Q*Nₜ
      Vecₙ[:,:,q] = RBVars.S.Φₛᵘ' * Vector(DEIM_mat_new[:, q])
    end
    Vecₙ = reshape(RBVars.S.Φₛᵘ' * reshape(MatqΦ,RBVars.S.Nₛᵘ,:),
      RBVars.S.nₛᵘ,:,Q*Nₜ)
  else
    Q = size(DEIM_mat)[2]
    Vecₙ = zeros(RBVars.S.nₛᵘ,1,Q)
    for q = 1:Q
      Vecₙ[:,:,q] = RBVars.S.Φₛᵘ' * Vector(DEIM_mat[:, q])
    end
    Vecₙ = reshape(Vecₙ,:,Q)
  end

  if var == "F"
    RBVars.S.Fₙ = Vecₙ
    RBVars.S.Qᶠ = Q
  elseif var == "H"
    RBVars.S.Hₙ = Vecₙ
    RBVars.S.Qʰ = Q
  else
    error("Unrecognized vector to assemble with DEIM")
  end

end

function assemble_offline_structures(RBInfo::Info, RBVars::PoissonSTGRB, operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.S.offline_time += @elapsed begin
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

function save_affine_structures(RBInfo::Info, RBVars::PoissonSTGRB)

  if RBInfo.save_offline_structures
    save_CSV(reshape(RBVars.Mₙ, :, RBVars.Qᵐ),
      joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    save_affine_structures(RBInfo, RBVars.S)
  end

end

function get_affine_structures(RBInfo::Info, RBVars::PoissonSTGRB) :: Vector
  operators = String[]
  append!(operators, get_Mₙ(RBInfo, RBVars))
  append!(operators, get_affine_structures(RBInfo, RBVars.S))
  return operators
end

function get_Q(RBInfo::Info, RBVars::PoissonSTGRB)
  if RBVars.Qᵐ == 0
    RBVars.Qᵐ = size(RBVars.Mₙ)[end]
  end
  get_Q(RBInfo, RBVars.S)
end

function get_RB_LHS_blocks(RBInfo, RBVars::PoissonSTGRB, θᵐ, θᵃ)

  @info "Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)"

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ
  nₜᵘ = RBVars.nₜᵘ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.S.Qᵃ

  Φₜᵘ_M = zeros(RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ₁_M = zeros(RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ_A = zeros(RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)
  Φₜᵘ₁_A = zeros(RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐ[q,:])
    for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end])
    for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵃ[q,:])
    for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end])
    for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  Mₙ_tmp = zeros(RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Mₙ₁_tmp = zeros(RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Aₙ_tmp = zeros(RBVars.nᵘ,RBVars.nᵘ,Qᵃ)
  Aₙ₁_tmp = zeros(RBVars.nᵘ,RBVars.nᵘ,Qᵃ)
  for qᵐ = 1:Qᵐ
    Mₙ_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ_M[:,:,qᵐ])
    Mₙ₁_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ₁_M[:,:,qᵐ])
  end
  for qᵃ = 1:Qᵃ
    Aₙ_tmp[:,:,qᵃ] = kron(RBVars.S.Aₙ[:,:,qᵃ],Φₜᵘ_A[:,:,qᵃ])
    Aₙ₁_tmp[:,:,qᵃ] = kron(RBVars.S.Aₙ[:,:,qᵃ],Φₜᵘ₁_A[:,:,qᵃ])
  end
  Mₙ = reshape(sum(Mₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Mₙ₁ = reshape(sum(Mₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ = δtθ*reshape(sum(Aₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ₁ = δtθ*reshape(sum(Aₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)

  block₁ = θ*(Aₙ+Mₙ) + (1-θ)*Aₙ₁ - θ*Mₙ₁
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

  initialize_RB_system(RBVars.S)
  initialize_online_time(RBVars.S)

  RBVars.S.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    blocks = [1]
    operators = get_system_blocks(RBInfo,RBVars,blocks,blocks)

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
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ)
      else
        build_Param_RHS(RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,blocks,blocks,operators)

end

function build_Param_RHS(RBInfo::Info, RBVars::PoissonSTGRB, Param)
  @info "Assembling RHS exactly using θ-method time scheme, θ=$(RBInfo.θ)"
  δtθ = RBInfo.δt*RBInfo.θ
  F_t = assemble_forcing(FEMSpace, RBInfo, Param)
  H_t = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  F, H = zeros(RBVars.S.Nₛᵘ, RBVars.Nₜ), zeros(RBVars.S.Nₛᵘ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo)
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

function adaptive_cycle_offline(
  RBInfo::Info,
  RBVars::PoissonUnsteady,
  mean_pointwise_err::Matrix,
  n_adaptive=[5,5])

  sᵢ = argmax(abs.(mean_pointwise_err),dims=1)
  tᵢ = argmax(abs.(mean_pointwise_err),dims=2)
  ind_s = [sᵢ[i][1] for i=1:n_adaptive[1]]
  ind_t = [tᵢ[i][2] for i=1:n_adaptive[2]]
  Sᵘ = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"),
      DataFrame))[:, 1:RBVars.Nₜ]
  Φₛᵘ_new = Sᵘ[:,ind_s]
  Φₜᵘ_new = Sᵘ[ind_t,:]
  Φₛᵘ_old = RBVars.Φₛᵘ
  Φₜᵘ_old = RBVars.Φₜᵘ
  RBVars.Φₛᵘ = Φₛᵘ_new
  RBVars.Φₜᵘ = Φₜᵘ_new
  assemble_offline_structures(RBInfo, RBVars)
  RBVars.Φₛᵘ = hcat(Φₛᵘ_old,RBVars.Φₛᵘ)
  RBVars.Φₜᵘ = hcat(Φₜᵘ_old,RBVars.Φₜᵘ)

  loop_on_params(RBInfo,RBVars,μ,param_nbs)

end
