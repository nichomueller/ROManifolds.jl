function get_Aₙ(ROM_info::Problem, RB_variables::PoissonSGRB) :: Vector

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
    @info "Importing reduced affine stiffness matrix"
    Aₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
    RB_variables.Aₙ = reshape(Aₙ,RB_variables.nₛᵘ,RB_variables.nₛᵘ,:)
    RB_variables.Qᵃ = size(RB_variables.Aₙ)[3]
    return []
  else
    @info "Failed to import the reduced affine stiffness matrix: must build it"
    return ["A"]
  end

end

function assemble_affine_matrices(ROM_info::Problem, RB_variables::PoissonSGRB, var::String)

  if var === "A"
    RB_variables.Qᵃ = 1
    @info "Assembling affine reduced stiffness"
    A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
    RB_variables.Aₙ = zeros(RB_variables.nₛᵘ, RB_variables.nₛᵘ, RB_variables.Qᵃ)
    RB_variables.Aₙ[:,:,1] = (RB_variables.Φₛᵘ)' * A * RB_variables.Φₛᵘ
  else
    @error "Unrecognized variable to load"
  end

end

function assemble_MDEIM_matrices(ROM_info::Problem, RB_variables::PoissonSGRB, var::String)

  if var === "A"

    @info "The stiffness is non-affine: running the MDEIM offline phase on $(ROM_info.nₛ_MDEIM) snapshots"
    MDEIM_mat, RB_variables.MDEIM_idx_A, RB_variables.sparse_el_A, MDEIM_err_bound, MDEIM_Σ = MDEIM_offline(problem_info, ROM_info, "A")
    RB_variables.Qᵃ = size(MDEIM_mat)[2]
    RB_variables.Aₙ = zeros(RB_variables.nₛᵘ, RB_variables.nₛᵘ, RB_variables.Qᵃ)
    for q = 1:RB_variables.Qᵃ
      @info "S-GRB: affine component number $q, matrix $var"
      Aq = reshape(MDEIM_mat[:, q], (RB_variables.Nₛᵘ, RB_variables.Nₛᵘ))
      RB_variables.Aₙ[:,:,q] = RB_variables.Φₛᵘ' * Matrix(Aq) * RB_variables.Φₛᵘ
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

function assemble_affine_vectors(ROM_info::Problem, RB_variables::PoissonSGRB, var::String)

  if var === "F"
    RB_variables.Qᶠ = 1
    @info "Assembling affine reduced forcing term"
    F = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "F.csv"))
    RB_variables.Fₙ = (RB_variables.Φₛᵘ)' * F
  elseif var === "H"
    RB_variables.Qʰ = 1
    @info "Assembling affine reduced Neumann term"
    H = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "H.csv"))
    RB_variables.Hₙ = (RB_variables.Φₛᵘ)' * H
  else
    @error "Unrecognized variable to load"
  end

end

function assemble_DEIM_vectors(ROM_info::Problem, RB_variables::PoissonSGRB, var::String)

  @info "S-GRB: running the DEIM offline phase on variable $var with $nₛ_DEIM snapshots"

  DEIM_mat, DEIM_idx, _, _ = DEIM_offline(problem_info, ROM_info, var)
  DEIMᵢ_mat = Matrix(DEIM_mat[DEIM_idx, :])
  Q = size(DEIM_mat)[2]
  varₙ = zeros(RB_variables.nₛᵘ,Q)
  for q = 1:Q
    varₙ[:,q] = RB_variables.Φₛᵘ' * Vector(DEIM_mat[:, q])
  end

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

function save_affine_structures(ROM_info::Problem, RB_variables::PoissonSGRB)

  if ROM_info.save_offline_structures

    Aₙ = reshape(RB_variables.Aₙ, :, RB_variables.Qᵃ)
    save_CSV(Aₙ, joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
    save_CSV([RB_variables.Qᵃ], joinpath(ROM_info.paths.ROM_structures_path, "Qᵃ.csv"))

    if !ROM_info.build_parametric_RHS
      save_CSV(RB_variables.Fₙ, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
      save_CSV([RB_variables.Qᶠ], joinpath(ROM_info.paths.ROM_structures_path, "Qᶠ.csv"))
      save_CSV(RB_variables.Hₙ, joinpath(ROM_info.paths.ROM_structures_path, "Hₙ.csv"))
      save_CSV([RB_variables.Qʰ], joinpath(ROM_info.paths.ROM_structures_path, "Qʰ.csv"))
    end

  end

end

function build_param_RHS(ROM_info::Problem, RB_variables::PoissonSGRB, param, θᵃ) :: Tuple

  FE_space = get_FE_space(problem_info, param.model)
  F, H = assemble_forcing(FE_space, ROM_info, param)
  Fₙ, Hₙ = (RB_variables.Φₛᵘ)' * F, (RB_variables.Φₛᵘ)' * H

  reshape(Fₙ, :, 1), reshape(Hₙ, :, 1)

end

function get_θ(ROM_info::Problem, RB_variables::PoissonSGRB, param) :: Tuple

  θᵃ = get_θᵃ(ROM_info, RB_variables, param)
  if !ROM_info.build_parametric_RHS
    θᶠ, θʰ = get_θᶠʰ(ROM_info, RB_variables, param)
  else
    θᶠ, θʰ = Float64[], Float64[]
  end

  return θᵃ, θᶠ, θʰ

end
