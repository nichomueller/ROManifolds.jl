
function get_inverse_P_matrix(ROM_info::Info, RB_variables::PoissonSPGRB)

  if use_norm_X

    if isempty(RB_variables.Pᵤ⁻¹) || maximum(abs.(RB_variables.Pᵤ⁻¹)) === 0
      @info "S-PGRB: building the inverse of the diag preconditioner of the H1 norm matrix"

      if isfile(joinpath(ROM_info.paths.FEM_structures_path, "Pᵤ⁻¹.csv"))
        RB_variables.Pᵤ⁻¹ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Pᵤ⁻¹.csv"); convert_to_sparse = true)
      else
        get_norm_matrix(ROM_info, RB_variables)
        diag_Xᵘ₀ = Vector(diag(RB_variables.Xᵘ₀))
        RB_variables.Pᵤ⁻¹ = spdiagm(1 ./ diag_Xᵘ₀)
        save_CSV(RB_variables.Pᵤ⁻¹, joinpath(ROM_info.paths.FEM_structures_path, "Pᵤ⁻¹.csv"))
      end

    end

    RB_variables.Pᵤ⁻¹ = Matrix(RB_variables.Pᵤ⁻¹)

  else

    RB_variables.Pᵤ⁻¹ = I(RB_variables.Nₛᵘ)

  end

end

function get_Aₙ(ROM_info::Info, RB_variables::PoissonSPGRB) :: Vector

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    @info "Importing reduced affine stiffness matrix"
    Aₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
    RB_variables.Aₙ = reshape(Aₙ,RB_variables.nₛᵘ,RB_variables.nₛᵘ,:)
    @info "S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹"
    AΦᵀPᵤ⁻¹ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    RB_variables.AΦᵀPᵤ⁻¹ = reshape(AΦᵀPᵤ⁻¹,RB_variables.nₛᵘ,RB_variables.Nₛᵘ,:)
    RB_variables.Qᵃ = size(RB_variables.Aₙ)[3]
    return []
  else
    @info "Failed to import the reduced affine stiffness matrix: must build it"
    return ["A"]
  end

end

function get_AΦᵀPᵤ⁻¹(ROM_info::Info, RB_variables::PoissonSPGRB)

  @info "S-PGRB: fetching the matrix AΦᵀPᵤ⁻¹"
  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    AΦᵀPᵤ⁻¹ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    RB_variables.AΦᵀPᵤ⁻¹ = reshape(AΦᵀPᵤ⁻¹,RB_variables.nₛᵘ,RB_variables.Nₛᵘ,:)
    return
  else
    if !ROM_info.probl_nl["A"]
      @info "S-PGRB: failed to build AΦᵀPᵤ⁻¹; have to assemble affine stiffness"
      assemble_affine_matrices(ROM_info, RB_variables, "A")
    else
      @info "S-PGRB: failed to build AΦᵀPᵤ⁻¹; have to assemble non-affine stiffness "
      assemble_MDEIM_matrices(ROM_info, RB_variables, "A")
    end
  end

end

function assemble_affine_matrices(ROM_info::Info, RB_variables::PoissonSPGRB, var::String)

  get_inverse_P_matrix(ROM_info, RB_variables)

  if var === "A"
    RB_variables.Qᵃ = 1
    @info "Assembling affine reduced stiffness"
    @info "SPGRB: affine component number 1, matrix A"
    A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
    RB_variables.AΦᵀPᵤ⁻¹ = zeros(RB_variables.nₛᵘ, RB_variables.Nₛᵘ, RB_variables.Qᵃ)
    RB_variables.Aₙ = zeros(RB_variables.nₛᵘ, RB_variables.nₛᵘ, RB_variables.Qᵃ^2)
    RB_variables.AΦᵀPᵤ⁻¹[:,:,1] = (A * RB_variables.Φₛᵘ)' * RB_variables.Pᵤ⁻¹
    RB_variables.Aₙ[:,:,1] = RB_variables.AΦᵀPᵤ⁻¹[:,:,1] * (A * RB_variables.Φₛᵘ)
  else
    @error "Unrecognized variable to load"
  end

end

function assemble_MDEIM_matrices(ROM_info::Info, RB_variables::PoissonSPGRB, var::String)

  get_inverse_P_matrix(ROM_info, RB_variables)

  if var === "A"

    @info "The stiffness is non-affine: running the MDEIM offline phase on $(ROM_info.nₛ_MDEIM) snapshots. This might take some time"
    MDEIM_mat, RB_variables.MDEIM_idx_A, RB_variables.sparse_el_A, MDEIM_err_bound, MDEIM_Σ = MDEIM_offline(FE_space, ROM_info, "A")
    RB_variables.Qᵃ = size(MDEIM_mat)[2]

    AΦ = zeros(RB_variables.Nₛᵘ, RB_variables.nₛᵘ, RB_variables.Qᵃ)
    RB_variables.Aₙ = zeros(RB_variables.nₛᵘ, RB_variables.nₛᵘ, RB_variables.Qᵃ^2)
    RB_variables.AΦᵀPᵤ⁻¹ = zeros(RB_variables.nₛᵘ, RB_variables.Nₛᵘ, RB_variables.Qᵃ)
    for q = 1:RB_variables.Qᵃ
      AΦ[:,:,q] = reshape(Vector(MDEIM_mat[:, q]), RB_variables.Nₛᵘ, RB_variables.Nₛᵘ) * RB_variables.Φₛᵘ
    end
    tensor_product(RB_variables.AΦᵀPᵤ⁻¹, AΦ, RB_variables.Pᵤ⁻¹, transpose_A=true)

    for q₁ = 1:RB_variables.Qᵃ
      for q₂ = 1:RB_variables.Qᵃ
        @info "SPGRB: affine component number $((q₁-1)*RB_variables.Qᵃ+q₂), matrix A"
        RB_variables.Aₙ[:, :, (q₁-1)*RB_variables.Qᵃ+q₂] = RB_variables.AΦᵀPᵤ⁻¹[:, :, q₁] * AΦ[:, :, q₂]
      end
    end
    RB_variables.MDEIMᵢ_A = Matrix(MDEIM_mat[RB_variables.MDEIM_idx_A, :])

    if ROM_info.save_offline_structures
      save_CSV([MDEIM_err_bound], joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_err_bound.csv"))
      save_CSV(MDEIM_Σ, joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    end

  else

    @error "Unrecognized variable to load"

  end

end

function assemble_affine_vectors(ROM_info::Info, RB_variables::PoissonSPGRB, var::String)

  @info "S-PGRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots"

  if var === "F"
    RB_variables.Qᶠ = 1
    @info "Assembling affine reduced forcing term"
    F = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "F.csv"))
    Fₙ = zeros(RB_variables.nₛᵘ, 1, RB_variables.Qᵃ*RB_variables.Qᶠ)
    tensor_product(Fₙ, RB_variables.AΦᵀPᵤ⁻¹, reshape(F,:,1))
    RB_variables.Fₙ = reshape(Fₙ,:,RB_variables.Qᵃ*RB_variables.Qᶠ)
  elseif var === "H"
    RB_variables.Qʰ = 1
    @info "Assembling affine reduced Neumann term"
    H = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "H.csv"))
    Hₙ = zeros(RB_variables.nₛᵘ, 1, RB_variables.Qᵃ*RB_variables.Qʰ)
    tensor_product(Hₙ, RB_variables.AΦᵀPᵤ⁻¹, reshape(H,:,1))
    RB_variables.Hₙ = reshape(Hₙ,:,RB_variables.Qᵃ*RB_variables.Qʰ)
  else
    @error "Unrecognized variable to assemble"
  end

end

function assemble_DEIM_vectors(ROM_info::Info, RB_variables::PoissonSPGRB, var::String)

  @info "SPGRB: forcing term is non-affine: running the DEIM offline phase on $nₛ_DEIM snapshots; A is affine"

  DEIM_mat, DEIM_idx, _, _ = DEIM_offline(FE_space, ROM_info, var)
  DEIMᵢ_mat = Matrix(DEIM_mat[DEIM_idx, :])
  Q = size(DEIM_mat)[2]
  varₙ = zeros(RB_variables.nₛᵘ,1,RB_variables.Qᵃ*Q)
  tensor_product(varₙ,RB_variables.AΦᵀPᵤ⁻¹,DEIM_mat)
  varₙ = reshape(varₙ,:,RB_variables.Qᵃ*Q)

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
    @error "Unrecognized variable to assemble"
  end

end

function save_affine_structures(ROM_info::Info, RB_variables::PoissonSPGRB)

  if ROM_info.save_offline_structures

    Aₙ = reshape(RB_variables.Aₙ, :, RB_variables.Qᵃ)
    AΦᵀPᵤ⁻¹ = reshape(RB_variables.AΦᵀPᵤ⁻¹, :, RB_variables.Qᵃ)
    save_CSV(Aₙ, joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
    save_CSV(AΦᵀPᵤ⁻¹, joinpath(ROM_info.paths.ROM_structures_path, "AΦᵀPᵤ⁻¹.csv"))
    save_CSV([RB_variables.Qᵃ], joinpath(ROM_info.paths.ROM_structures_path, "Qᵃ.csv"))

    if !ROM_info.build_parametric_RHS
      save_CSV(RB_variables.Fₙ, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
      save_CSV([RB_variables.Qᶠ], joinpath(ROM_info.paths.ROM_structures_path, "Qᶠ.csv"))
      save_CSV(RB_variables.Hₙ, joinpath(ROM_info.paths.ROM_structures_path, "Hₙ.csv"))
      save_CSV([RB_variables.Qʰ], joinpath(ROM_info.paths.ROM_structures_path, "Qʰ.csv"))
    end

  end

end

function build_param_RHS(ROM_info::Info, RB_variables::PoissonSPGRB, param, θᵃ::Array) :: Tuple

  θᵃ_temp = θᵃ[1:RB_variables.Qᵃ]/sqrt(θᵃ[1])
  F = assemble_forcing(FE_space, ROM_info, param)
  H = assemble_neumann_datum(FE_space, ROM_info, param)
  AΦᵀPᵤ⁻¹ = assemble_online_structure(θᵃ_temp, RB_variables.AΦᵀPᵤ⁻¹)
  Fₙ, Hₙ = AΦᵀPᵤ⁻¹ * F, AΦᵀPᵤ⁻¹ * H

  reshape(Fₙ, :, 1), reshape(Hₙ, :, 1)

end

function get_θ(ROM_info::Info, RB_variables::PoissonSPGRB, param) :: Tuple

  θᵃ_temp = get_θᵃ(ROM_info, RB_variables, param)
  θᵃ = [θᵃ_temp[q₁]*θᵃ_temp[q₂] for q₁ = 1:RB_variables.Qᵃ for q₂ = 1:RB_variables.Qᵃ]

  if !ROM_info.build_parametric_RHS

    θᶠ_temp, θʰ_temp = get_θᶠʰ(ROM_info, RB_variables, param)
    θᶠ = [θᵃ_temp[q₁]*θᶠ_temp[q₂] for q₁ = 1:RB_variables.Qᵃ for q₂ = 1:RB_variables.Qᶠ]
    θʰ = [θᵃ_temp[q₁]*θʰ_temp[q₂] for q₁ = 1:RB_variables.Qᵃ for q₂ = 1:RB_variables.Qʰ]

  else

    θᶠ, θʰ = Float64[], Float64[]

  end

  θᵃ, θᶠ, θʰ

end
