include("S-PGRB_Poisson.jl")

function get_Aₙ(ROM_info::Problem, RB_variables::PoissonSTPGRB) :: Vector

  get_Aₙ(ROM_info, RB_variables.steady_info)

end

function get_Mₙ(ROM_info::Problem, RB_variables::PoissonSTPGRB) :: Vector

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "MΦ.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "MΦᵀPᵤ⁻¹.csv"))
    @info "Importing reduced affine stiffness matrix"
    Mₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    RB_variables.Mₙ = reshape(Mₙ,RB_variables.steady_info.nₛᵘ,RB_variables.steady_info.nₛᵘ,:)
    MΦ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MΦ.csv"))
    RB_variables.MΦ = reshape(MΦ,RB_variables.steady_info.Nₛᵘ,RB_variables.steady_info.nₛᵘ,:)
    MΦᵀPᵤ⁻¹ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MΦᵀPᵤ⁻¹.csv"))
    RB_variables.MΦᵀPᵤ⁻¹ = reshape(MΦᵀPᵤ⁻¹,RB_variables.nₛᵘ,RB_variables.Nₛᵘ,:)
    RB_variables.Qᵐ = size(RB_variables.Mₙ)[3]
    return []
  else
    @info "Failed to import the reduced affine mass matrix: must build it"
    return ["M"]
  end

end

function get_MAₙ(ROM_info::Problem, RB_variables::PoissonSTPGRB) :: Vector

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    @info "S-PGRB: importing MAₙ"
    MAₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MAₙ.csv"))
    RB_variables.MAₙ = reshape(MAₙ,RB_variables.nₛᵘ,RB_variables.nₛᵘ,:)
    return []
  else
    @info "ST-PGRB: failed to import MAₙ: must build it"
    return ["MA"]
  end

end

function assemble_MAₙ(RB_variables::PoissonSTPGRB)

  @info "S-PGRB: Assembling MAₙ"

  nₛᵘ = RB_variables.steady_info.nₛᵘ
  MAₙ = zeros(RB_variables.steady_info.nₛᵘ,RB_variables.steady_info.nₛᵘ,RB_variables.Qᵐ*RB_variables.Qᵃ)
  [MAₙ[:,:,(i-1)*nₛᵘ+j] = RB_variables.MΦ'[:,:,i] * RB_variables.steady_info.AΦᵀPᵤ⁻¹'[:,:,j] for i=1:nₛᵘ for j=1:nₛᵘ]
  RB_variables.MAₙ = MAₙ

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

  @info "The matrix $var is non-affine: running the MDEIM offline phase on $nₛ_MDEIM snapshots"
  MDEIM_mat, MDEIM_idx, sparse_el, _, _ = MDEIM_offline(problem_info, ROM_info, var)
  Q = size(MDEIM_mat)[2]
  MDEIMᵢ_mat = Matrix(MDEIM_mat[MDEIM_idx, :])

  if var === "M"

    RB_variables.MΦᵀPᵤ⁻¹ = zeros(RB_variables.steady_info.nₛᵘ, RB_variables.steady_info.Nₛᵘ, RB_variables.Qᵐ)
    RB_variables.Mₙ = zeros(RB_variables.steady_info.nₛᵘ, RB_variables.steady_info.nₛᵘ, RB_variables.Qᵐ^2)
    RB_variables.MΦ = zeros(RB_variables.steady_info.Nₛᵘ, RB_variables.steady_info.nₛᵘ, RB_variables.Qᵐ)
    for q = 1:RB_variables.Qᵐ
      RB_variables.MΦ[:,:,q] = reshape(Vector(MDEIM_mat[:, q]), RB_variables.steady_info.Nₛᵘ, RB_variables.steady_info.Nₛᵘ) * RB_variables.steady_info.Φₛᵘ
    end
    tensor_product(RB_variables.MΦᵀPᵤ⁻¹, MΦ, RB_variables.Pᵤ⁻¹, transpose_A=true)

    for q₁ = 1:RB_variables.Qᵐ
      for q₂ = 1:RB_variables.Qᵐ
        @info "ST-PGRB: affine component number $((q₁-1)*RB_variables.Qᵐ+q₂), matrix M"
        RB_variables.Mₙ[:, :, (q₁-1)*RB_variables.Qᵐ+q₂] = RB_variables.MΦᵀPᵤ⁻¹[:, :, q₁] * RB_variables.MΦ[:, :, q₂]
      end
    end

    RB_variables.MDEIMᵢ_M = MDEIMᵢ_mat
    RB_variables.MDEIM_idx_M = MDEIM_idx
    RB_variables.sparse_el_M = sparse_el
    RB_variables.Qᵐ = Q

  elseif var === "A"

    AΦ = zeros(RB_variables.Nₛᵘ, RB_variables.nₛᵘ, RB_variables.Qᵃ)
    RB_variables.Aₙ = zeros(RB_variables.nₛᵘ, RB_variables.nₛᵘ, RB_variables.Qᵃ^2)
    RB_variables.AΦᵀPᵤ⁻¹ = zeros(RB_variables.nₛᵘ, RB_variables.Nₛᵘ, RB_variables.Qᵃ)
    for q = 1:RB_variables.Qᵃ
      AΦ[:,:,q] = reshape(Vector(MDEIM_mat[:, q]), RB_variables.Nₛᵘ, RB_variables.Nₛᵘ) * RB_variables.Φₛᵘ
    end
    tensor_product(RB_variables.AΦᵀPᵤ⁻¹, AΦ, RB_variables.Pᵤ⁻¹, transpose_A=true)

    for q₁ = 1:RB_variables.Qᵃ
      for q₂ = 1:RB_variables.Qᵃ
        @info "ST-PGRB: affine component number $((q₁-1)*RB_variables.Qᵃ+q₂), matrix A"
        RB_variables.Aₙ[:, :, (q₁-1)*RB_variables.Qᵃ+q₂] = RB_variables.AΦᵀPᵤ⁻¹[:, :, q₁] * AΦ[:, :, q₂]
      end
    end

    RB_variables.steady_info.MDEIMᵢ_A = MDEIMᵢ_mat
    RB_variables.steady_info.MDEIM_idx_A = MDEIM_idx
    RB_variables.steady_info.sparse_el_A = sparse_el
    RB_variables.steady_info.Qᵃ = Q

  else

    @error "Unrecognized variable to assemble with MDEIM"

  end

end

function assemble_affine_vectors(ROM_info::Problem, RB_variables::PoissonSTPGRB, var::String)

  @info "SPGRB: assembling affine reduced RHS; A is affine"

  if var === "F"
    RB_variables.Qᶠ = 1
    @info "Assembling affine reduced forcing term"
    F = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "F.csv"))
    MFₙ = zeros(RB_variables.nₛᵘ, 1, RB_variables.Qᵐ*RB_variables.Qᶠ)
    tensor_product(MFₙ, RB_variables.MΦᵀPᵤ⁻¹, reshape(F,:,1))
    AFₙ = zeros(RB_variables.nₛᵘ, 1, RB_variables.Qᵃ*RB_variables.Qᶠ)
    tensor_product(AFₙ, RB_variables.AΦᵀPᵤ⁻¹, reshape(F,:,1))
    RB_variables.Fₙ = hcat(reshape(MFₙ,:,RB_variables.Qᵐ*RB_variables.Qᶠ), reshape(AFₙ,:,RB_variables.Qᵃ*RB_variables.Qᶠ))
  elseif var === "H"
    RB_variables.Qʰ = 1
    @info "Assembling affine reduced Neumann term"
    H = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "H.csv"))
    MHₙ = zeros(RB_variables.nₛᵘ, 1, RB_variables.Qᵐ*RB_variables.Qʰ)
    tensor_product(MHₙ, RB_variables.MΦᵀPᵤ⁻¹, reshape(H,:,1))
    AHₙ = zeros(RB_variables.nₛᵘ, 1, RB_variables.Qᵃ*RB_variables.Qʰ)
    tensor_product(AHₙ, RB_variables.AΦᵀPᵤ⁻¹, reshape(H,:,1))
    RB_variables.Hₙ = hcat(reshape(MHₙ,:,RB_variables.Qᵐ*RB_variables.Qʰ), reshape(AHₙ,:,RB_variables.Qᵃ*RB_variables.Qʰ))
  else
    @error "Unrecognized variable to assemble"
  end

end

function assemble_DEIM_vectors(ROM_info::Problem, RB_variables::PoissonSTPGRB, var::String)

  @info "ST-PGRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots"

  DEIM_mat, DEIM_idx, _, _ = DEIM_offline(problem_info, ROM_info, var)
  DEIMᵢ_mat = Matrix(DEIM_mat[DEIM_idx, :])
  Q = size(DEIM_mat)[2]
  Mvarₙ = zeros(RB_variables.nₛᵘ,1,RB_variables.Qᵐ*Q)
  tensor_product(Mvarₙ,RB_variables.MΦᵀPᵤ⁻¹,DEIM_mat)
  Mvarₙ = reshape(Mvarₙ,:,RB_variables.Qᵐ*Q)
  Avarₙ = zeros(RB_variables.nₛᵘ,1,RB_variables.Qᵃ*Q)
  tensor_product(Avarₙ,RB_variables.AΦᵀPᵤ⁻¹,DEIM_mat)
  Avarₙ = reshape(Avarₙ,:,RB_variables.Qᵃ*Q)

  if var === "F"
    RB_variables.DEIMᵢ_mat_F = DEIMᵢ_mat
    RB_variables.DEIM_idx_F = DEIM_idx
    RB_variables.Qᶠ = Q
    RB_variables.Fₙ = hcat(Mvarₙ,Avarₙ)
  elseif var === "H"
    RB_variables.DEIMᵢ_mat_H = DEIMᵢ_mat
    RB_variables.DEIM_idx_H = DEIM_idx
    RB_variables.Qʰ = Q
    RB_variables.Hₙ = hcat(Mvarₙ,Avarₙ)
  else
    @error "Unrecognized variable to assemble"
  end

end

function assemble_offline_structures(ROM_info::Problem, RB_variables::PoissonSTPGRB, operators=nothing)

  if isnothing(operators)
    operatorsₜ = set_operators(ROM_info, RB_variables)
  end

  assembly_time = 0
  if "M" ∈ operatorsₜ || "MA" ∈ operatorsₜ || "F" ∈ operators || "H" ∈ operators
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

  if "A" ∈ operators || "MA" ∈ operatorsₜ || "F" ∈ operators || "H" ∈ operators
    if !ROM_info.probl_nl["A"]
      assembly_time += @elapsed begin
        assemble_affine_matrices(ROM_info, RB_variables, "A")
      end
    else
      assembly_time += @elapsed begin
        assemble_MDEIM_matrices(ROM_info, RB_variables, "A")
      end
    end
  end

  if "F" ∈ operators
    if !ROM_info.probl_nl["f"]
      assembly_time += @elapsed begin
        assemble_affine_vectors(ROM_info, RB_variables, "F")
      end
    else
      assembly_time += @elapsed begin
        assemble_DEIM_vectors(ROM_info, RB_variables, "F")
      end
    end
  end

  if "H" ∈ operators
    if !ROM_info.probl_nl["h"]
      assembly_time += @elapsed begin
        assemble_affine_vectors(ROM_info, RB_variables, "H")
      end
    else
      assembly_time += @elapsed begin
        assemble_DEIM_vectors(ROM_info, RB_variables, "H")
      end
    end
  end

  assemble_MAₙ(RB_variables)

  RB_variables.steady_info.offline_time += assembly_time
  save_affine_structures(ROM_info, RB_variables)
  save_M_DEIM_structures(ROM_info, RB_variables)

end

function save_affine_structures(ROM_info::Problem, RB_variables::PoissonSTPGRB)

  if ROM_info.save_offline_structures
    Mₙ = reshape(RB_variables.Mₙ, :, RB_variables.Qᵐ^2)
    MAₙ = reshape(RB_variables.MAₙ, :, RB_variables.Qᵐ*RB_variables.Qᵃ)
    MΦᵀPᵤ⁻¹ = reshape(RB_variables.MΦᵀPᵤ⁻¹, :, RB_variables.Qᵐ)
    save_CSV(Mₙ, joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    save_CSV(MAₙ, joinpath(ROM_info.paths.ROM_structures_path, "MAₙ.csv"))
    save_CSV(MΦᵀPᵤ⁻¹, joinpath(ROM_info.paths.ROM_structures_path, "MΦᵀPᵤ⁻¹.csv"))
    save_CSV([RB_variables.Qᵐ], joinpath(ROM_info.paths.ROM_structures_path, "Qᵐ.csv"))
    save_affine_structures(ROM_info, RB_variables)
  end

end

function get_affine_structures(ROM_info::Problem, RB_variables::PoissonSTPGRB) :: Vector

  operators = String[]

  append!(operators, get_affine_structures(ROM_info, RB_variables.steady_info))
  append!(operators, get_Mₙ(ROM_info, RB_variables))
  append!(operators, get_MAₙ(ROM_info, RB_variables))

  return operators

end

function get_RB_LHS_blocks(ROM_info, RB_variables::PoissonSTPGRB, θᵐ, θᵃ, θᵐᵃ, θᵃᵐ)

  @info "Assembling LHS using Crank-Nicolson time scheme"

  θ = ROM_info.θ
  δt = ROM_info.δt
  nₜᵘ = RB_variables.nₜᵘ
  Qᵐ = RB_variables.Qᵐ
  Qᵃ = RB_variables.steady_info.Qᵃ
  Qᵐᵃ = RB_variables.Qᵐᵃ

  Φₜᵘ_M = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐ)
  Φₜᵘ₋₁₋₁_M = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐ)
  Φₜᵘ₁₋₁_M = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐ)

  Φₜᵘ_A = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵃ)
  Φₜᵘ₋₁₋₁_A = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵃ)
  Φₜᵘ₁₋₁_A = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵃ)

  Φₜᵘ_MA = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₋₁₋₁_MA = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₁₋₁_MA = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐᵃ)

  Φₜᵘ_AM = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₋₁₋₁_AM = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐᵃ)
  Φₜᵘ₁₋₁_AM = zeros(RB_variables.nₜᵘ, RB_variables.nₜᵘ, Qᵐᵃ)

  [Φₜᵘ_M[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*RB_variables.Φₜᵘ[:,j_t].*θᵐ[q,:]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_A[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*RB_variables.Φₜᵘ[:,j_t].*θᵃ[q,:]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_MA[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*RB_variables.Φₜᵘ[:,j_t].*θᵐᵃ[q,:]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ_AM[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*RB_variables.Φₜᵘ[:,j_t].*θᵃᵐ[q,:]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  [Φₜᵘ₋₁₋₁_M[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₋₁_A[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₋₁_MA[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵐᵃ[q,2:end]) for q = 1:QQᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₋₁_AM[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵃᵐ[q,2:end]) for q = 1:QQᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  [Φₜᵘ₋₁₁_M[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*RB_variables.Φₜᵘ[2:end,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_A[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*RB_variables.Φₜᵘ[2:end,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_MA[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*RB_variables.Φₜᵘ[2:end,j_t].*θᵐᵃ[q,2:end]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_AM[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*RB_variables.Φₜᵘ[2:end,j_t].*θᵃᵐ[q,2:end]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  [Φₜᵘ₁₋₁_M[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[2:end,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end]) for q = 1:Qᵐ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁₋₁_A[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[2:end,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end]) for q = 1:Qᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁₋₁_MA[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[2:end,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵐᵃ[q,2:end]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]
  [Φₜᵘ₁₋₁_AM[i_t,j_t,q] = sum(RB_variables.Φₜᵘ[2:end,i_t].*RB_variables.Φₜᵘ[1:end-1,j_t].*θᵃᵐ[q,2:end]) for q = 1:Qᵐᵃ for i_t = 1:nₜᵘ for j_t = 1:nₜᵘ]

  #check: Φₜᵘ₋₁₁_M === (Φₜᵘ₁₋₁_M)ᵀ

  block₁ = zeros(RB_variables.nᵘ, RB_variables.nᵘ)

  for i_s = 1:RB_variables.steady_info.nₛᵘ

    for i_t = 1:RB_variables.nₜᵘ

      i_st = index_mapping(i_s, i_t, RB_variables)

      for j_s = 1:RB_variables.steady_info.nₛᵘ
        for j_t = 1:RB_variables.nₜᵘ

          j_st = index_mapping(j_s, j_t, RB_variables)

          term1 = δt^2*RB_variables.steady_info.Aₙ[i_s,j_s,:]'*( θ^2*Φₜᵘ_A[i_t,j_t,:] + (1-θ)^2*θ^2*Φₜᵘ₋₁₋₁_A[i_t,j_t,:] + θ*(1-θ)*Φₜᵘ₋₁₁_A[i_t,j_t,:] + θ*(1-θ)*Φₜᵘ₁₋₁_A[i_t,j_t,:] )
          term2 = δt*RB_variables.MAₙ[i_s,j_s,:]'*( θ*Φₜᵘ_MA[i_t,j_t,:] - (1-θ)*Φₜᵘ₋₁₋₁_MA[i_t,j_t,:] - θ*Φₜᵘ₋₁₁_MA[i_t,j_t,:] + (1-θ)*Φₜᵘ₁₋₁_MA[i_t,j_t,:] )
          term3 = δt*RB_variables.MAₙ[i_s,j_s,:]'*( θ*Φₜᵘ_AM[i_t,j_t,:] - (1-θ)*Φₜᵘ₋₁₋₁_AM[i_t,j_t,:] + (1-θ)*Φₜᵘ₋₁₁_AM[i_t,j_t,:] - θ*Φₜᵘ₁₋₁_AM[i_t,j_t,:] )
          term4 = RB_variables.M[i_s,j_s,:]'*( Φₜᵘ_M[i_t,j_t,:] + Φₜᵘ₋₁₋₁_M[i_t,j_t,:] - Φₜᵘ₋₁₁_M[i_t,j_t,:] - Φₜᵘ₁₋₁_AM[i_t,j_t,:])

          block₁[i_st,j_st] = θ^2*(term1 + term2 + term3 + term4)

        end
      end

    end
  end

  push!(RB_variables.steady_info.LHSₙ, block₁)

end

function get_RB_RHS_blocks(ROM_info::Problem, RB_variables::PoissonSTPGRB, θᶠ, θʰ)

  @info "Assembling RHS"

  Qᵐᶠ = RB_variables.Qᵐ*RB_variables.steady_info.Qᶠ
  Qᵐʰ = RB_variables.Qᵐ*RB_variables.steady_info.Qʰ
  Qᶠ_tot = RB_variables.steady_info.Qᶠ*(RB_variables.Qᵐ+RB_variables.steady_info.Qᵃ)
  Qʰ_tot = RB_variables.steady_info.Qʰ*(RB_variables.Qᵐ+RB_variables.steady_info.Qᵃ)
  δt = ROM_info.δt
  θ = ROM_info.θ
  δtθ = δt*θ
  nₜᵘ = RB_variables.nₜᵘ

  Φₜᵘ_F = zeros(RB_variables.nₜᵘ, Qᶠ*Qᵐ)
  Φₜᵘ₋₁₁_F = zeros(RB_variables.nₜᵘ, Qᶠ*Qᵐ)
  Φₜᵘ_H = zeros(RB_variables.nₜᵘ, Qʰ*Qᵐ)
  Φₜᵘ₋₁₁_H = zeros(RB_variables.nₜᵘ, Qʰ*Qᵐ)

  [Φₜᵘ_F[i_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*θᶠ[q,:]) for q = 1:Qᶠ_tot for i_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_F[i_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*θᶠ[q,2:end]) for q = 1:Qᶠ_tot for i_t = 1:nₜᵘ]
  [Φₜᵘ_H[i_t,q] = sum(RB_variables.Φₜᵘ[:,i_t].*θʰ[q,:]) for q = 1:Qʰ_tot for i_t = 1:nₜᵘ]
  [Φₜᵘ₋₁₁_H[i_t,q] = sum(RB_variables.Φₜᵘ[1:end-1,i_t].*θʰ[q,2:end]) for q = 1:Qʰ_tot for i_t = 1:nₜᵘ]

  block₁ = zeros(RB_variables.nᵘ,1)
  for i_s = 1:RB_variables.steady_info.nₛᵘ
    for i_t = 1:RB_variables.nₜᵘ

      i_st = index_mapping(i_s, i_t, RB_variables)

      term1 = RB_variables.steady_info.Fₙ[i_s,1:Qᵐᶠ]'*Φₜᵘ_F[i_t,1:Qᵐᶠ] + RB_variables.steady_info.Hₙ[i_s,1:Qᵐʰ]'*Φₜᵘ_H[i_t,1:Qᵐʰ]
      term2 = δtθ*(RB_variables.steady_info.Fₙ[i_s,Qᵐᶠ+1:end]'*Φₜᵘ_F[i_t,Qᵐᶠ+1:end] + RB_variables.steady_info.Hₙ[i_s,Qᵐʰ+1:end]'*Φₜᵘ_H[i_t,Qᵐʰ+1:end])
      term3 = -θ*( RB_variables.steady_info.Fₙ[i_s,1:Qᵐᶠ]'*Φₜᵘ₋₁₁_F[i_t,1:Qᵐᶠ] + RB_variables.steady_info.Hₙ[i_s,1:Qᵐʰ]'*Φₜᵘ₋₁₁_H[i_t,1:Qᵐʰ] )
      term4 = δtθ*(1-θ)*(RB_variables.steady_info.Fₙ[i_s,Qᵐᶠ+1:end]'*Φₜᵘ₋₁₁_F[i_t,Qᵐᶠ+1:end] + RB_variables.steady_info.Hₙ[i_s,Qᵐʰ+1:end]'*Φₜᵘ₋₁₁_H[i_t,Qᵐʰ+1:end])

      block₁[i_st] = term1 + term2 + term3 + term4

    end
  end

  block₁ *= δtθ
  push!(RB_variables.steady_info.RHSₙ, reshape(block₁, :, 1))

end

function get_RB_system(ROM_info::Problem, RB_variables::PoissonSTPGRB, param)

  @info "Preparing the RB system: fetching reduced LHS"
  initialize_RB_system(RB_variables.steady_info)
  get_Q(ROM_info, RB_variables)
  blocks = [1]
  operators = get_system_blocks(ROM_info, RB_variables, blocks, blocks)

  θᵐ, θᵃ, θᶠ, θʰ = get_θ(ROM_info, RB_variables, param)

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

function build_param_RHS(ROM_info::Problem, RB_variables::PoissonSTPGRB, param, θᵐ, θᵃ)

  δt = ROM_info.δt
  θ = ROM_info.θ
  δtθ = δt*θ
  θᵐ_temp = θᵐ[1:RB_variables.Qᵐ]/sqrt(θᵐ[1])
  θᵃ_temp = θᵃ[1:RB_variables.steady_info.Qᵃ]/sqrt(θᵃ[1])

  FE_space = get_FE_space(problem_info, param.model)
  F_t, H_t = assemble_forcing(FE_space, RB_variables, param)
  F, H = zeros(RB_variables.steady_info.Nₛᵘ, RB_variables.Nₜ), zeros(RB_variables.steady_info.Nₛᵘ, RB_variables.Nₜ)
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+δtθ
  for (i, tᵢ) in enumerate(times_θ)
    F[:,i] = F_t(tᵢ)
    H[:,i] = H_t(tᵢ)
  end
  RHS = (F+H)*δtθ

  MΦᵀPᵤ⁻¹ = assemble_online_structure(θᵐ_temp, RB_variables.MΦᵀPᵤ⁻¹)
  AΦᵀPᵤ⁻¹ = assemble_online_structure(θᵃ_temp, RB_variables.steady_info.AΦᵀPᵤ⁻¹)
  RHSΦₜ = RHS*RB_variables.Φₜᵘ
  RHSΦₜ₁ = RHS[:,2:end]*RB_variables.Φₜᵘ[1:end-1,:]

  RHSₙ = (MΦᵀPᵤ⁻¹+δtθ*AΦᵀPᵤ⁻¹)*RHSΦₜ + θ*(δt*(1-θ)*AΦᵀPᵤ⁻¹-MΦᵀPᵤ⁻¹)*RHSΦₜ₁

  push!(RB_variables.steady_info.RHSₙ, reshape(RHS,:,1))

end

function get_θ(ROM_info::Problem, RB_variables::PoissonSTPGRB, param) :: Tuple

  θᵐ_temp = get_θᵐ(ROM_info, RB_variables, param)
  θᵃ_temp = get_θᵃ(ROM_info, RB_variables, param)

  Qᵐ, Qᵃ = length(θᵐ_temp), length(θᵃ_temp)
  θᵐ = [θᵐ_temp[q₁]*θᵐ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qᵐ]
  θᵐᵃ = [θᵐ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:Qᵐ for q₂ = 1:Qᵃ]
  θᵃᵐ = [θᵐ_temp[q₁]*θᵃ_temp[q₂] for q₂ = 1:Qᵃ for q₁ = 1:Qᵐ]

  θᵃ = [θᵃ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:Qᵃ for q₂ = 1:Qᵃ]

  if !ROM_info.build_parametric_RHS

    θᶠ_temp, θʰ_temp = get_θᶠʰ(ROM_info, RB_variables, param)
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
