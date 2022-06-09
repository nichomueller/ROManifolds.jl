function get_Aₙ(RBInfo::Info, RBVars::PoissonSGRB) :: Vector

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv"))
    @info "Importing reduced affine stiffness matrix"
    Aₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv"))
    RBVars.Aₙ = reshape(Aₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)
    RBVars.Qᵃ = size(RBVars.Aₙ)[3]
    return []
  else
    @info "Failed to import the reduced affine stiffness matrix: must build it"
    return ["A"]
  end

end

function assemble_affine_matrices(RBInfo::Info, RBVars::PoissonSGRB, var::String)

  if var == "A"
    RBVars.Qᵃ = 1
    @info "Assembling affine reduced stiffness"
    A = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
    RBVars.Aₙ = zeros(RBVars.nₛᵘ, RBVars.nₛᵘ, RBVars.Qᵃ)
    RBVars.Aₙ[:,:,1] = (RBVars.Φₛᵘ)' * A * RBVars.Φₛᵘ
  else
    error("Unrecognized variable to load")
  end

end

function assemble_MDEIM_matrices(RBInfo::Info, RBVars::PoissonSGRB, var::String)

  if var == "A"

    @info "The stiffness is non-affine: running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots"
    MDEIM_mat, RBVars.MDEIM_idx_A, RBVars.sparse_el_A, MDEIM_err_bound, MDEIM_Σ = MDEIM_offline(FEMSpace, RBInfo, "A")
    RBVars.Qᵃ = size(MDEIM_mat)[2]
    RBVars.Aₙ = zeros(RBVars.nₛᵘ, RBVars.nₛᵘ, RBVars.Qᵃ)
    for q = 1:RBVars.Qᵃ
      @info "S-GRB: affine component number $q, matrix $var"
      Aq = reshape(MDEIM_mat[:, q], (RBVars.Nₛᵘ, RBVars.Nₛᵘ))
      RBVars.Aₙ[:,:,q] = RBVars.Φₛᵘ' * Matrix(Aq) * RBVars.Φₛᵘ
    end
    RBVars.MDEIMᵢ_A = Matrix(MDEIM_mat[RBVars.MDEIM_idx_A, :])
    if RBInfo.save_offline_structures
      save_CSV([MDEIM_err_bound], joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_err_bound.csv"))
      save_CSV(MDEIM_Σ, joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    end

  else

    error("Unrecognized variable to load")

  end

end

function assemble_affine_vectors(RBInfo::Info, RBVars::PoissonSGRB, var::String)

  if var == "F"
    RBVars.Qᶠ = 1
    @info "Assembling affine reduced forcing term"
    F = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "F.csv"))
    RBVars.Fₙ = (RBVars.Φₛᵘ)' * F
  elseif var == "H"
    RBVars.Qʰ = 1
    @info "Assembling affine reduced Neumann term"
    H = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "H.csv"))
    RBVars.Hₙ = (RBVars.Φₛᵘ)' * H
  else
    error("Unrecognized variable to load")
  end

end

function assemble_DEIM_vectors(RBInfo::Info, RBVars::PoissonSGRB, var::String)

  @info "S-GRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots"

  DEIM_mat, DEIM_idx, _, _ = DEIM_offline(FEMSpace, RBInfo, var)
  DEIMᵢ_mat = Matrix(DEIM_mat[DEIM_idx, :])
  Q = size(DEIM_mat)[2]
  varₙ = zeros(RBVars.nₛᵘ,Q)
  for q = 1:Q
    varₙ[:,q] = RBVars.Φₛᵘ' * Vector(DEIM_mat[:, q])
  end

  if var == "F"
    RBVars.DEIMᵢ_mat_F = DEIMᵢ_mat
    RBVars.DEIM_idx_F = DEIM_idx
    RBVars.Qᶠ = Q
    RBVars.Fₙ = varₙ
  elseif var == "H"
    RBVars.DEIMᵢ_mat_H = DEIMᵢ_mat
    RBVars.DEIM_idx_H = DEIM_idx
    RBVars.Qʰ = Q
    RBVars.Hₙ = varₙ
  else
    error("Unrecognized vector to assemble with DEIM")
  end

end

function save_affine_structures(RBInfo::Info, RBVars::PoissonSGRB)

  if RBInfo.save_offline_structures

    Aₙ = reshape(RBVars.Aₙ, :, RBVars.Qᵃ)
    save_CSV(Aₙ, joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv"))

    if !RBInfo.build_Parametric_RHS
      save_CSV(RBVars.Fₙ, joinpath(RBInfo.paths.ROM_structures_path, "Fₙ.csv"))
      save_CSV(RBVars.Hₙ, joinpath(RBInfo.paths.ROM_structures_path, "Hₙ.csv"))
    end

  end

end

function build_Param_RHS(RBInfo::Info, RBVars::PoissonSGRB, Param, ::Array) ::Tuple

  F = assemble_forcing(FEMSpace, RBInfo, Param)
  H = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  Fₙ, Hₙ = (RBVars.Φₛᵘ)' * F, (RBVars.Φₛᵘ)' * H

  reshape(Fₙ, :, 1), reshape(Hₙ, :, 1)

end

function get_θ(RBInfo::Info, RBVars::PoissonSGRB, Param) ::Tuple

  θᵃ = get_θᵃ(RBInfo, RBVars, Param)
  if !RBInfo.build_Parametric_RHS
    θᶠ, θʰ = get_θᶠʰ(RBInfo, RBVars, Param)
  else
    θᶠ, θʰ = Float64[], Float64[]
  end

  return θᵃ, θᶠ, θʰ

end
