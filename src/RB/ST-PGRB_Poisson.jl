function get_Aₙ(RBInfo::Info, RBVars::PoissonSTPGRB) :: Vector

  get_Aₙ(RBInfo, RBVars.S)

end

function get_Mₙ(RBInfo::Info, RBVars::PoissonSTPGRB) :: Vector

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv")) && isfile(joinpath(RBInfo.paths.ROM_structures_path, "MΦ.csv")) && isfile(joinpath(RBInfo.paths.ROM_structures_path, "MΦᵀPᵤ⁻¹.csv"))
    @info "Importing reduced affine stiffness matrix"
    Mₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    RBVars.Mₙ = reshape(Mₙ,RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,:)
    MΦ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MΦ.csv"))
    RBVars.MΦ = reshape(MΦ,RBVars.S.Nₛᵘ,RBVars.S.nₛᵘ,:)
    MΦᵀPᵤ⁻¹ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MΦᵀPᵤ⁻¹.csv"))
    RBVars.MΦᵀPᵤ⁻¹ = reshape(MΦᵀPᵤ⁻¹,RBVars.nₛᵘ,RBVars.Nₛᵘ,:)
    RBVars.Qᵐ = size(RBVars.Mₙ)[3]
    return []
  else
    @info "Failed to import the reduced affine mass matrix: must build it"
    return ["M"]
  end

end

function get_MAₙ(RBInfo::Info, RBVars::PoissonSTPGRB) :: Vector

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    @info "S-PGRB: importing MAₙ"
    MAₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MAₙ.csv"))
    RBVars.MAₙ = reshape(MAₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)
    return []
  else
    @info "ST-PGRB: failed to import MAₙ: must build it"
    return ["MA"]
  end

end

function assemble_MAₙ(RBVars::PoissonSTPGRB)

  @info "S-PGRB: Assembling MAₙ"

  nₛᵘ = RBVars.S.nₛᵘ
  MAₙ = zeros(RBVars.S.nₛᵘ,RBVars.S.nₛᵘ,RBVars.Qᵐ*RBVars.Qᵃ)
  [MAₙ[:,:,(i-1)*nₛᵘ+j] = RBVars.MΦ'[:,:,i] * RBVars.S.AΦᵀPᵤ⁻¹'[:,:,j] for i=1:nₛᵘ for j=1:nₛᵘ]
  RBVars.MAₙ = MAₙ

end

function assemble_affine_matrices(RBInfo::Info, RBVars::PoissonSTPGRB, var::String)

  get_inverse_P_matrix(RBInfo, RBVars)

  if var == "M"
    RBVars.Qᵐ = 1
    @info "Assembling affine reduced mass"
    M = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "M.csv"); convert_to_sparse = true)
    RBVars.Mₙ = zeros(RBVars.S.nₛᵘ, RBVars.S.nₛᵘ, RBVars.Qᵐ)
    RBVars.Mₙ[:,:,1] = (M*RBVars.S.Φₛᵘ)' * RBVars.Pᵤ⁻¹ * (M*RBVars.S.Φₛᵘ)
    RBVars.MΦ = zeros(RBVars.S.Nₛᵘ, RBVars.S.nₛᵘ, RBVars.Qᵐ)
    RBVars.MΦ[:,:,1] = M*RBVars.S.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.S, var)
  end

end

function assemble_MDEIM_matrices(RBInfo::Info, RBVars::PoissonSTPGRB, var::String)

  @info "The matrix $var is non-affine: running the MDEIM offline phase on $nₛ_MDEIM snapshots"
  MDEIM_mat, MDEIM_idx, sparse_el, _, _ = MDEIM_offline(FEMSpace, RBInfo, var)
  Q = size(MDEIM_mat)[2]
  MDEIMᵢ_mat = Matrix(MDEIM_mat[MDEIM_idx, :])

  if var == "M"

    RBVars.MΦᵀPᵤ⁻¹ = zeros(RBVars.S.nₛᵘ, RBVars.S.Nₛᵘ, RBVars.Qᵐ)
    RBVars.Mₙ = zeros(RBVars.S.nₛᵘ, RBVars.S.nₛᵘ, RBVars.Qᵐ^2)
    RBVars.MΦ = zeros(RBVars.S.Nₛᵘ, RBVars.S.nₛᵘ, RBVars.Qᵐ)
    for q = 1:RBVars.Qᵐ
      RBVars.MΦ[:,:,q] = reshape(Vector(MDEIM_mat[:, q]), RBVars.S.Nₛᵘ, RBVars.S.Nₛᵘ) * RBVars.S.Φₛᵘ
    end
    matrix_product!(RBVars.MΦᵀPᵤ⁻¹, MΦ, RBVars.Pᵤ⁻¹, transpose_A=true)

    for q₁ = 1:RBVars.Qᵐ
      for q₂ = 1:RBVars.Qᵐ
        @info "ST-PGRB: affine component number $((q₁-1)*RBVars.Qᵐ+q₂), matrix M"
        RBVars.Mₙ[:, :, (q₁-1)*RBVars.Qᵐ+q₂] = RBVars.MΦᵀPᵤ⁻¹[:, :, q₁] * RBVars.MΦ[:, :, q₂]
      end
    end

    RBVars.MDEIMᵢ_M = MDEIMᵢ_mat
    RBVars.MDEIM_idx_M = MDEIM_idx
    RBVars.sparse_el_M = sparse_el
    RBVars.Qᵐ = Q

  elseif var == "A"

    AΦ = zeros(RBVars.Nₛᵘ, RBVars.nₛᵘ, RBVars.Qᵃ)
    RBVars.Aₙ = zeros(RBVars.nₛᵘ, RBVars.nₛᵘ, RBVars.Qᵃ^2)
    RBVars.AΦᵀPᵤ⁻¹ = zeros(RBVars.nₛᵘ, RBVars.Nₛᵘ, RBVars.Qᵃ)
    for q = 1:RBVars.Qᵃ
      AΦ[:,:,q] = reshape(Vector(MDEIM_mat[:, q]), RBVars.Nₛᵘ, RBVars.Nₛᵘ) * RBVars.Φₛᵘ
    end
    matrix_product!(RBVars.AΦᵀPᵤ⁻¹, AΦ, RBVars.Pᵤ⁻¹, transpose_A=true)

    for q₁ = 1:RBVars.Qᵃ
      for q₂ = 1:RBVars.Qᵃ
        @info "ST-PGRB: affine component number $((q₁-1)*RBVars.Qᵃ+q₂), matrix A"
        RBVars.Aₙ[:, :, (q₁-1)*RBVars.Qᵃ+q₂] = RBVars.AΦᵀPᵤ⁻¹[:, :, q₁] * AΦ[:, :, q₂]
      end
    end

    RBVars.S.MDEIMᵢ_A = MDEIMᵢ_mat
    RBVars.S.MDEIM_idx_A = MDEIM_idx
    RBVars.S.sparse_el_A = sparse_el
    RBVars.S.Qᵃ = Q

  else

    error("Unrecognized variable to assemble with MDEIM")

  end

end

function assemble_affine_vectors(RBInfo::Info, RBVars::PoissonSTPGRB, var::String)

  @info "SPGRB: assembling affine reduced RHS; A is affine"

  if var == "F"
    RBVars.Qᶠ = 1
    @info "Assembling affine reduced forcing term"
    F = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "F.csv"))
    MFₙ = zeros(RBVars.nₛᵘ, 1, RBVars.Qᵐ*RBVars.Qᶠ)
    matrix_product!(MFₙ, RBVars.MΦᵀPᵤ⁻¹, reshape(F,:,1))
    AFₙ = zeros(RBVars.nₛᵘ, 1, RBVars.Qᵃ*RBVars.Qᶠ)
    matrix_product!(AFₙ, RBVars.AΦᵀPᵤ⁻¹, reshape(F,:,1))
    RBVars.Fₙ = hcat(reshape(MFₙ,:,RBVars.Qᵐ*RBVars.Qᶠ), reshape(AFₙ,:,RBVars.Qᵃ*RBVars.Qᶠ))
  elseif var == "H"
    RBVars.Qʰ = 1
    @info "Assembling affine reduced Neumann term"
    H = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "H.csv"))
    MHₙ = zeros(RBVars.nₛᵘ, 1, RBVars.Qᵐ*RBVars.Qʰ)
    matrix_product!(MHₙ, RBVars.MΦᵀPᵤ⁻¹, reshape(H,:,1))
    AHₙ = zeros(RBVars.nₛᵘ, 1, RBVars.Qᵃ*RBVars.Qʰ)
    matrix_product!(AHₙ, RBVars.AΦᵀPᵤ⁻¹, reshape(H,:,1))
    RBVars.Hₙ = hcat(reshape(MHₙ,:,RBVars.Qᵐ*RBVars.Qʰ), reshape(AHₙ,:,RBVars.Qᵃ*RBVars.Qʰ))
  else
    error("Unrecognized variable to assemble")
  end

end

function assemble_DEIM_vectors(RBInfo::Info, RBVars::PoissonSTPGRB, var::String)

  @info "ST-PGRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots"

  DEIM_mat, DEIM_idx, _, _ = DEIM_offline(FEMSpace, RBInfo, var)
  DEIMᵢ_mat = Matrix(DEIM_mat[DEIM_idx, :])
  Q = size(DEIM_mat)[2]
  Mvarₙ = zeros(RBVars.nₛᵘ,1,RBVars.Qᵐ*Q)
  matrix_product!(Mvarₙ,RBVars.MΦᵀPᵤ⁻¹,DEIM_mat)
  Mvarₙ = reshape(Mvarₙ,:,RBVars.Qᵐ*Q)
  Avarₙ = zeros(RBVars.nₛᵘ,1,RBVars.Qᵃ*Q)
  matrix_product!(Avarₙ,RBVars.AΦᵀPᵤ⁻¹,DEIM_mat)
  Avarₙ = reshape(Avarₙ,:,RBVars.Qᵃ*Q)

  if var == "F"
    RBVars.DEIMᵢ_mat_F = DEIMᵢ_mat
    RBVars.DEIM_idx_F = DEIM_idx
    RBVars.Qᶠ = Q
    RBVars.Fₙ = hcat(Mvarₙ,Avarₙ)
  elseif var == "H"
    RBVars.DEIMᵢ_mat_H = DEIMᵢ_mat
    RBVars.DEIM_idx_H = DEIM_idx
    RBVars.Qʰ = Q
    RBVars.Hₙ = hcat(Mvarₙ,Avarₙ)
  else
    error("Unrecognized variable to assemble")
  end

end

function assemble_offline_structures(RBInfo::Info, RBVars::PoissonSTPGRB, operators=nothing)

  if isnothing(operators)
    operatorsₜ = set_operators(RBInfo, RBVars)
  end

  assembly_time = 0
  if "A" ∈ operators || "MA" ∈ operatorsₜ || "F" ∈ operators || "H" ∈ operators
    assembly_time += @elapsed begin
      if !RBInfo.probl_nl["M"]
        assemble_affine_matrices(RBInfo, RBVars, "M")
      else
        assemble_MDEIM_matrices(RBInfo, RBVars, "M")
      end
    end
  end

  if "A" ∈ operators || "MA" ∈ operatorsₜ || "F" ∈ operators || "H" ∈ operators
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

  assemble_MAₙ(RBVars)

  RBVars.S.offline_time += assembly_time
  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function save_affine_structures(RBInfo::Info, RBVars::PoissonSTPGRB)

  if RBInfo.save_offline_structures
    Mₙ = reshape(RBVars.Mₙ, :, RBVars.Qᵐ^2)
    MAₙ = reshape(RBVars.MAₙ, :, RBVars.Qᵐ*RBVars.Qᵃ)
    MΦᵀPᵤ⁻¹ = reshape(RBVars.MΦᵀPᵤ⁻¹, :, RBVars.Qᵐ)
    save_CSV(Mₙ, joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    save_CSV(MAₙ, joinpath(RBInfo.paths.ROM_structures_path, "MAₙ.csv"))
    save_CSV(MΦᵀPᵤ⁻¹, joinpath(RBInfo.paths.ROM_structures_path, "MΦᵀPᵤ⁻¹.csv"))
    save_affine_structures(RBInfo, RBVars)
  end

end

function get_affine_structures(RBInfo::Info, RBVars::PoissonSTPGRB) :: Vector

  operators = String[]

  append!(operators, get_affine_structures(RBInfo, RBVars.S))
  append!(operators, get_Mₙ(RBInfo, RBVars))
  append!(operators, get_MAₙ(RBInfo, RBVars))

  return operators

end

function get_RB_LHS_blocks(RBInfo, RBVars::PoissonSTPGRB, θᵐ, θᵃ, θᵐᵃ, θᵃᵐ)

  @info "Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)"

  θ = RBInfo.θ
  δt = RBInfo.δt
  nₜᵘ = RBVars.nₜᵘ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.S.Qᵃ
  Qᵐᵃ = RBVars.Qᵐᵃ

  Φₜᵘ_M = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)
  Φₜᵘ₋₁₋₁_M = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)
  Φₜᵘ₁₋₁_M = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐ)

  Φₜᵘ_A = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)
  Φₜᵘ₋₁₋₁_A = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)
  Φₜᵘ₁₋₁_A = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵃ)

  Φₜᵘ_MA = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₋₁₋₁_MA = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₁₋₁_MA = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)

  Φₜᵘ_AM = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₋₁₋₁_AM = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₁₋₁_AM = zeros(RBVars.nₜᵘ, RBVars.nₜᵘ, Qᵐᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐ[q,:]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵃ[q,:]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_MA[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐᵃ[q,:]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_AM[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵃᵐ[q,:]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  [Φₜᵘ₋₁₋₁_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₋₁_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₋₁_MA[i_t,j_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐᵃ[q,2:end]) for q = 1:QQᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₋₁_AM[i_t,j_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃᵐ[q,2:end]) for q = 1:QQᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  [Φₜᵘ₋₁₁_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[2:end,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[2:end,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_MA[i_t,j_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[2:end,j_t].*θᵐᵃ[q,2:end]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_AM[i_t,j_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*RBVars.Φₜᵘ[2:end,j_t].*θᵃᵐ[q,2:end]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  [Φₜᵘ₁₋₁_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁₋₁_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁₋₁_MA[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐᵃ[q,2:end]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁₋₁_AM[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃᵐ[q,2:end]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  #check: Φₜᵘ₋₁₁_M == (Φₜᵘ₁₋₁_M)ᵀ

  block₁ = zeros(RBVars.nᵘ, RBVars.nᵘ)

  for i_s = 1:RBVars.S.nₛᵘ

    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      for j_s = 1:RBVars.S.nₛᵘ
        for j_t = 1:RBVars.nₜᵘ

          j_st = index_mapping(j_s, j_t, RBVars)

          term1 = δt^2*RBVars.S.Aₙ[i_s,j_s,:]'*( θ^2*Φₜᵘ_A[i_t,j_t,:] + (1-θ)^2*θ^2*Φₜᵘ₋₁₋₁_A[i_t,j_t,:] + θ*(1-θ)*Φₜᵘ₋₁₁_A[i_t,j_t,:] + θ*(1-θ)*Φₜᵘ₁₋₁_A[i_t,j_t,:] )
          term2 = δt*RBVars.MAₙ[i_s,j_s,:]'*( θ*Φₜᵘ_MA[i_t,j_t,:] - (1-θ)*Φₜᵘ₋₁₋₁_MA[i_t,j_t,:] - θ*Φₜᵘ₋₁₁_MA[i_t,j_t,:] + (1-θ)*Φₜᵘ₁₋₁_MA[i_t,j_t,:] )
          term3 = δt*RBVars.MAₙ[i_s,j_s,:]'*( θ*Φₜᵘ_AM[i_t,j_t,:] - (1-θ)*Φₜᵘ₋₁₋₁_AM[i_t,j_t,:] + (1-θ)*Φₜᵘ₋₁₁_AM[i_t,j_t,:] - θ*Φₜᵘ₁₋₁_AM[i_t,j_t,:] )
          term4 = RBVars.M[i_s,j_s,:]'*( Φₜᵘ_M[i_t,j_t,:] + Φₜᵘ₋₁₋₁_M[i_t,j_t,:] - Φₜᵘ₋₁₁_M[i_t,j_t,:] - Φₜᵘ₁₋₁_AM[i_t,j_t,:])

          block₁[i_st,j_st] = θ^2*(term1 + term2 + term3 + term4)

        end
      end

    end
  end

  push!(RBVars.S.LHSₙ, block₁)

end

function get_RB_RHS_blocks(RBInfo::Info, RBVars::PoissonSTPGRB, θᶠ, θʰ)

  @info "Assembling RHS using θ-method time scheme, θ=$(RBInfo.θ)"

  Qᵐᶠ = RBVars.Qᵐ*RBVars.S.Qᶠ
  Qᵐʰ = RBVars.Qᵐ*RBVars.S.Qʰ
  Qᶠ_tot = RBVars.S.Qᶠ*(RBVars.Qᵐ+RBVars.S.Qᵃ)
  Qʰ_tot = RBVars.S.Qʰ*(RBVars.Qᵐ+RBVars.S.Qᵃ)
  δt = RBInfo.δt
  θ = RBInfo.θ
  δtθ = δt*θ
  nₜᵘ = RBVars.nₜᵘ

  Φₜᵘ_F = zeros(RBVars.nₜᵘ, Qᶠ*Qᵐ)
  Φₜᵘ₋₁₁_F = zeros(RBVars.nₜᵘ, Qᶠ*Qᵐ)
  Φₜᵘ_H = zeros(RBVars.nₜᵘ, Qʰ*Qᵐ)
  Φₜᵘ₋₁₁_H = zeros(RBVars.nₜᵘ, Qʰ*Qᵐ)

  [Φₜᵘ_F[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θᶠ[q,:]) for q = 1:Qᶠ_tot for i_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_F[i_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*θᶠ[q,2:end]) for q = 1:Qᶠ_tot for i_t = 1:nₜᵘ]
  [Φₜᵘ_H[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θʰ[q,:]) for q = 1:Qʰ_tot for i_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_H[i_t,q] = sum(RBVars.Φₜᵘ[1:end-1,i_t].*θʰ[q,2:end]) for q = 1:Qʰ_tot for i_t = 1:nₜᵘ]

  block₁ = zeros(RBVars.nᵘ,1)
  for i_s = 1:RBVars.S.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t, RBVars)

      term1 = RBVars.S.Fₙ[i_s,1:Qᵐᶠ]'*Φₜᵘ_F[i_t,1:Qᵐᶠ] + RBVars.S.Hₙ[i_s,1:Qᵐʰ]'*Φₜᵘ_H[i_t,1:Qᵐʰ]
      term2 = δtθ*(RBVars.S.Fₙ[i_s,Qᵐᶠ+1:end]'*Φₜᵘ_F[i_t,Qᵐᶠ+1:end] + RBVars.S.Hₙ[i_s,Qᵐʰ+1:end]'*Φₜᵘ_H[i_t,Qᵐʰ+1:end])
      term3 = -θ*( RBVars.S.Fₙ[i_s,1:Qᵐᶠ]'*Φₜᵘ₋₁₁_F[i_t,1:Qᵐᶠ] + RBVars.S.Hₙ[i_s,1:Qᵐʰ]'*Φₜᵘ₋₁₁_H[i_t,1:Qᵐʰ] )
      term4 = δtθ*(1-θ)*(RBVars.S.Fₙ[i_s,Qᵐᶠ+1:end]'*Φₜᵘ₋₁₁_F[i_t,Qᵐᶠ+1:end] + RBVars.S.Hₙ[i_s,Qᵐʰ+1:end]'*Φₜᵘ₋₁₁_H[i_t,Qᵐʰ+1:end])

      block₁[i_st,1] = term1 + term2 + term3 + term4

    end
  end

  block₁ *= δtθ
  push!(RBVars.S.RHSₙ, block₁)

end

function get_RB_system(RBInfo::Info, RBVars::PoissonSTPGRB, Param)

  @info "Preparing the RB system: fetching reduced LHS"
  initialize_RB_system(RBVars.S)
  get_Q(RBInfo, RBVars)
  blocks = [1]
  operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

  θᵐ, θᵃ, θᶠ, θʰ = get_θ(RBInfo, RBVars, Param)

  if "LHS" ∈ operators
    get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ)
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

function build_Param_RHS(RBInfo::Info, RBVars::PoissonSTPGRB, Param, θᵐ, θᵃ)

  δt = RBInfo.δt
  θ = RBInfo.θ
  δtθ = δt*θ
  θᵐ_temp = θᵐ[1:RBVars.Qᵐ]/sqrt(θᵐ[1])
  θᵃ_temp = θᵃ[1:RBVars.S.Qᵃ]/sqrt(θᵃ[1])

  F_t = assemble_forcing(FEMSpace, RBInfo, Param)
  H_t = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  F, H = zeros(RBVars.S.Nₛᵘ, RBVars.Nₜ), zeros(RBVars.S.Nₛᵘ, RBVars.Nₜ)
  timesθ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+δtθ
  for (i, tᵢ) in enumerate(timesθ)
    F[:,i] = F_t(tᵢ)
    H[:,i] = H_t(tᵢ)
  end
  RHS = (F+H)*δtθ

  MΦᵀPᵤ⁻¹ = assemble_online_structure(θᵐ_temp, RBVars.MΦᵀPᵤ⁻¹)
  AΦᵀPᵤ⁻¹ = assemble_online_structure(θᵃ_temp, RBVars.S.AΦᵀPᵤ⁻¹)
  RHSΦₜ = RHS*RBVars.Φₜᵘ
  RHSΦₜ₁ = RHS[:,2:end]*RBVars.Φₜᵘ[1:end-1,:]

  RHSₙ = (MΦᵀPᵤ⁻¹+δtθ*AΦᵀPᵤ⁻¹)*RHSΦₜ + θ*(δt*(1-θ)*AΦᵀPᵤ⁻¹-MΦᵀPᵤ⁻¹)*RHSΦₜ₁

  push!(RBVars.S.RHSₙ, RHS)

end

function get_θ(RBInfo::Info, RBVars::PoissonSTPGRB, Param) ::Tuple

  θᵐ_temp = get_θᵐ(RBInfo, RBVars, Param)
  θᵃ_temp = get_θᵃ(RBInfo, RBVars, Param)

  Qᵐ, Qᵃ = length(θᵐ_temp), length(θᵃ_temp)
  θᵐ = [θᵐ_temp[q₁]*θᵐ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qᵐ]
  θᵐᵃ = [θᵐ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qᵃ]
  θᵃᵐ = [θᵐ_temp[q₁]*θᵃ_temp[q₂] for q₂ = 1:Qᵃ for q₁ = 1:Qᵐ]

  θᵃ = [θᵃ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qᵃ]

  if !RBInfo.build_Parametric_RHS

    θᶠ_temp, θʰ_temp = get_θᶠʰ(RBInfo, RBVars, Param)
    Qᶠ, Qʰ = length(θᶠ_temp), length(θʰ_temp)
    θᵐᶠ = [θᵐ_temp[q₁]*θᶠ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qᶠ]
    θᵐʰ = [θᵐ_temp[q₁]*θʰ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qʰ]
    θᵃᶠ = [θᵃ_temp[q₁]*θᶠ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qᶠ]
    θᵃʰ = [θᵃ_temp[q₁]*θʰ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qʰ]
    θᶠ = hcat(θᵐᶠ, θᵃᶠ)
    θʰ = hcat(θᵐʰ, θᵃʰ)

  else

    θᶠ, θʰ = Float64[], Float64[]

  end

  return θᵐ, θᵐᵃ, θᵃᵐ, θᵃ, θᶠ, θʰ

end
