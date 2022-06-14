
function get_inverse_P_matrix(RBInfo::Info, RBVars::PoissonSPGRB)

  if RBInfo.use_norm_X
    if isempty(RBVars.Pᵤ⁻¹)
      @info "S-PGRB: building the inverse of the diag preconditioner of the H1 norm matrix"
      if isfile(joinpath(RBInfo.paths.FEM_structures_path, "Pᵤ⁻¹.csv"))
        RBVars.Pᵤ⁻¹ = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "Pᵤ⁻¹.csv");
          convert_to_sparse = true)
      else
        get_norm_matrix(RBInfo, RBVars)
        diag_Xᵘ₀ = Vector(diag(RBVars.Xᵘ₀))
        RBVars.Pᵤ⁻¹ = spdiagm(1 ./ diag_Xᵘ₀)
        save_CSV(RBVars.Pᵤ⁻¹, joinpath(RBInfo.paths.FEM_structures_path, "Pᵤ⁻¹.csv"))
      end
    end
    RBVars.Pᵤ⁻¹ = Matrix(RBVars.Pᵤ⁻¹)
  else
    RBVars.Pᵤ⁻¹ = I(RBVars.Nₛᵘ)
  end
end

function get_Aₙ(RBInfo::Info, RBVars::PoissonSPGRB) :: Vector

  if (isfile(joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv")) &&
      isfile(joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv")))
    @info "Importing reduced affine stiffness matrix"
    Aₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv"))
    RBVars.Aₙ = reshape(Aₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)
    @info "S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹"
    AΦᵀPᵤ⁻¹ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    RBVars.AΦᵀPᵤ⁻¹ = reshape(AΦᵀPᵤ⁻¹,RBVars.nₛᵘ,RBVars.Nₛᵘ,:)
    RBVars.Qᵃ = size(RBVars.Aₙ)[3]
    return []
  else
    @info "Failed to import the reduced affine stiffness matrix: must build it"
    return ["A"]
  end

end

function get_AΦᵀPᵤ⁻¹(RBInfo::Info, RBVars::PoissonSPGRB)

  @info "S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹"
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    AΦᵀPᵤ⁻¹ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    RBVars.AΦᵀPᵤ⁻¹ = reshape(AΦᵀPᵤ⁻¹,RBVars.nₛᵘ,RBVars.Nₛᵘ,:)
    return
  else
    if !RBInfo.probl_nl["A"]
      @info "S-PGRB: failed to build AΦᵀPᵤ⁻¹; have to assemble affine stiffness"
      assemble_affine_matrices(RBInfo, RBVars, "A")
    else
      @info "S-PGRB: failed to build AΦᵀPᵤ⁻¹; have to assemble non-affine stiffness "
      assemble_MDEIM_matrices(RBInfo, RBVars, "A")
    end
  end

end

function assemble_affine_matrices(RBInfo::Info, RBVars::PoissonSPGRB, var::String)

  get_inverse_P_matrix(RBInfo, RBVars)

  if var == "A"
    RBVars.Qᵃ = 1
    @info "Assembling affine reduced stiffness"
    @info "SPGRB: affine component number 1, matrix A"
    A = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "A.csv");
      convert_to_sparse = true)
    RBVars.AΦᵀPᵤ⁻¹ = zeros(RBVars.nₛᵘ, RBVars.Nₛᵘ, RBVars.Qᵃ)
    RBVars.Aₙ = zeros(RBVars.nₛᵘ, RBVars.nₛᵘ, RBVars.Qᵃ^2)
    RBVars.AΦᵀPᵤ⁻¹[:,:,1] = (A * RBVars.Φₛᵘ)' * RBVars.Pᵤ⁻¹
    RBVars.Aₙ[:,:,1] = RBVars.AΦᵀPᵤ⁻¹[:,:,1] * (A * RBVars.Φₛᵘ)
  else
    error("Unrecognized variable to load")
  end

end

function assemble_MDEIM_matrices(RBInfo::Info, RBVars::PoissonSPGRB, var::String)

  get_inverse_P_matrix(RBInfo, RBVars)

  @info "The stiffness is non-affine: running the MDEIM offline phase
    on $(RBInfo.nₛ_MDEIM) snapshots. This might take some time"
  if isnothing(RBVars.S.MDEIM_mat_A) || maximum(RBVars.S.MDEIM_mat_A) == 0
    RBVars.MDEIM_mat_A, RBVars.MDEIM_idx_A, RBVars.sparse_el_A,
      MDEIM_err_bound, MDEIM_Σ = MDEIM_offline(FEMSpace, RBInfo, "A")
  end
  RBVars.Qᵃ = size(RBVars.MDEIM_mat_A)[2]

  AΦ = zeros(RBVars.Nₛᵘ, RBVars.nₛᵘ, RBVars.Qᵃ)
  RBVars.Aₙ = zeros(RBVars.nₛᵘ, RBVars.nₛᵘ, RBVars.Qᵃ^2)
  RBVars.AΦᵀPᵤ⁻¹ = zeros(RBVars.nₛᵘ, RBVars.Nₛᵘ, RBVars.Qᵃ)
  for q = 1:RBVars.Qᵃ
    AΦ[:,:,q] = reshape(Vector(RBVars.MDEIM_mat_A[:, q]),
      RBVars.Nₛᵘ, RBVars.Nₛᵘ) * RBVars.Φₛᵘ
  end
  matrix_product!(RBVars.AΦᵀPᵤ⁻¹, AΦ, Matrix(RBVars.Pᵤ⁻¹), transpose_A=true)

  for q₁ = 1:RBVars.Qᵃ
    for q₂ = 1:RBVars.Qᵃ
      @info "SPGRB: affine component number $((q₁-1)*RBVars.Qᵃ+q₂), matrix A"
      RBVars.Aₙ[:, :, (q₁-1)*RBVars.Qᵃ+q₂] =
        (RBVars.AΦᵀPᵤ⁻¹[:,:,q₁]*AΦ[:,:,q₂])
    end
  end
  RBVars.MDEIMᵢ_A = Matrix(RBVars.MDEIM_mat_A[RBVars.MDEIM_idx_A, :])

  if RBInfo.save_offline_structures
    save_CSV([MDEIM_err_bound],
      joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_err_bound.csv"))
    save_CSV(MDEIM_Σ, joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
  end

end

function assemble_affine_vectors(RBInfo::Info, RBVars::PoissonSPGRB, var::String)

  @info "S-PGRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots"

  if var == "F"
    RBVars.Qᶠ = 1
    @info "Assembling affine reduced forcing term"
    F = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "F.csv"))
    Fₙ = zeros(RBVars.nₛᵘ, 1, RBVars.Qᵃ*RBVars.Qᶠ)
    matrix_product!(Fₙ, RBVars.AΦᵀPᵤ⁻¹, reshape(F,:,1))
    RBVars.Fₙ = reshape(Fₙ,:,RBVars.Qᵃ*RBVars.Qᶠ)
  elseif var == "H"
    RBVars.Qʰ = 1
    @info "Assembling affine reduced Neumann term"
    H = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "H.csv"))
    Hₙ = zeros(RBVars.nₛᵘ, 1, RBVars.Qᵃ*RBVars.Qʰ)
    matrix_product!(Hₙ, RBVars.AΦᵀPᵤ⁻¹, reshape(H,:,1))
    RBVars.Hₙ = reshape(Hₙ,:,RBVars.Qᵃ*RBVars.Qʰ)
  else
    error("Unrecognized variable to assemble")
  end

end

function assemble_DEIM_vectors(RBInfo::Info, RBVars::PoissonSPGRB, var::String)

  @info "SPGRB: forcing term is non-affine: running the DEIM offline phase on $nₛ_DEIM snapshots; A is affine"

  DEIM_mat, DEIM_idx, _, _ = DEIM_offline(FEMSpace, RBInfo, var)
  DEIMᵢ_mat = Matrix(DEIM_mat[DEIM_idx, :])
  Q = size(DEIM_mat)[2]
  varₙ = zeros(RBVars.nₛᵘ,1,RBVars.Qᵃ*Q)
  matrix_product!(varₙ,RBVars.AΦᵀPᵤ⁻¹,DEIM_mat)
  varₙ = reshape(varₙ,:,RBVars.Qᵃ*Q)

  if var == "F"
    RBVars.DEIMᵢ_F = DEIMᵢ_mat
    RBVars.DEIM_idx_F = DEIM_idx
    RBVars.Qᶠ = Q
    RBVars.Fₙ = varₙ
  elseif var == "H"
    RBVars.DEIMᵢ_H = DEIMᵢ_mat
    RBVars.DEIM_idx_H = DEIM_idx
    RBVars.Qʰ = Q
    RBVars.Hₙ = varₙ
  else
    error("Unrecognized variable to assemble")
  end

end

function save_affine_structures(RBInfo::Info, RBVars::PoissonSPGRB)

  if RBInfo.save_offline_structures

    Aₙ = reshape(RBVars.Aₙ, :, RBVars.Qᵃ)
    AΦᵀPᵤ⁻¹ = reshape(RBVars.AΦᵀPᵤ⁻¹, :, RBVars.Qᵃ)
    save_CSV(Aₙ, joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv"))
    save_CSV(AΦᵀPᵤ⁻¹, joinpath(RBInfo.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))

    if !RBInfo.build_Parametric_RHS
      save_CSV(RBVars.Fₙ, joinpath(RBInfo.paths.ROM_structures_path, "Fₙ.csv"))
      save_CSV(RBVars.Hₙ, joinpath(RBInfo.paths.ROM_structures_path, "Hₙ.csv"))
    end

  end

end

function build_Param_RHS(RBInfo::Info, RBVars::PoissonSPGRB, Param, θᵃ::Vector) :: Tuple

  θᵃ_temp = θᵃ[1:RBVars.Qᵃ]/sqrt(θᵃ[1])
  F = assemble_forcing(FEMSpace, RBInfo, Param)
  H = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  AΦᵀPᵤ⁻¹ = assemble_online_structure(θᵃ_temp, RBVars.AΦᵀPᵤ⁻¹)
  Fₙ, Hₙ = AΦᵀPᵤ⁻¹ * F, AΦᵀPᵤ⁻¹ * H

  reshape(Fₙ, :, 1), reshape(Hₙ, :, 1)

end

function get_θ(RBInfo::Info, RBVars::PoissonSPGRB, Param) :: Tuple

  θᵃ_temp = get_θᵃ(RBInfo, RBVars, Param)
  θᵃ = [θᵃ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:RBVars.Qᵃ for q₂ = 1:RBVars.Qᵃ]

  if !RBInfo.build_Parametric_RHS

    θᶠ_temp, θʰ_temp = get_θᶠʰ(RBInfo, RBVars, Param)
    θᶠ = [θᵃ_temp[q₁]*θᶠ_temp[q₂] for q₁ = 1:RBVars.Qᵃ for q₂ = 1:RBVars.Qᶠ]
    θʰ = [θᵃ_temp[q₁]*θʰ_temp[q₂] for q₁ = 1:RBVars.Qᵃ for q₂ = 1:RBVars.Qʰ]

  else

    θᶠ, θʰ = Float64[], Float64[]

  end

  θᵃ, θᶠ, θʰ

end
