function get_Aₙ(RBInfo::Info, RBVars::PoissonSGRB) :: Vector

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv"))
    @info "Importing reduced affine stiffness matrix"
    Aₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Aₙ.csv"))
    RBVars.Aₙ = reshape(Aₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)
    RBVars.Qᵃ = size(RBVars.Aₙ)[3]
    return []
  else
    @info "Failed to import Aₙ: must build it"
    return ["A"]
  end

end

function assemble_affine_matrices(RBInfo::Info, RBVars::PoissonSGRB, var::String)

  if var == "A"
    RBVars.Qᵃ = 1
    @info "Assembling affine reduced stiffness"
    A = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "A.csv");
      convert_to_sparse = true)
    RBVars.Aₙ = reshape((RBVars.Φₛᵘ)'*A*RBVars.Φₛᵘ,RBVars.nₛᵘ,RBVars.nₛᵘ,1)
  else
    error("Unrecognized variable to load")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonSGRB,
  MDEIM_mat::Matrix,
  row_idx::Vector)

  Q = size(MDEIM_mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.S.Nₛᵘ)
  MatqΦ = zeros(RBVars.S.Nₛᵘ,RBVars.S.nₛᵘ,Q)
  @simd for j = 1:RBVars.S.Nₛᵘ
    Mat_idx = findall(x -> x == j, r_idx)
    MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.S.Φₛᵘ[c_idx[Mat_idx],:])'
  end
  RBVars.S.Aₙ = reshape(RBVars.S.Φₛᵘ' *
    reshape(MatqΦ,RBVars.S.Nₛᵘ,:),RBVars.S.nₛᵘ,:,Q)
  RBVars.S.Qᵃ = Q

end

function assemble_reduced_mat_DEIM(
  RBVars::PoissonSTGRB,
  DEIM_mat::Matrix,
  var::String)

  Q = size(DEIM_mat)[2]
  Vecₙ = zeros(RBVars.S.nₛᵘ,1,Q)
  @simd for q = 1:Q
    Vecₙ[:,:,q] = RBVars.S.Φₛᵘ' * Vector(DEIM_mat[:, q])
  end
  Vecₙ = reshape(Vecₙ,:,Q)

  if var == "F"
    RBVars.S.Fₙ = Vecₙ
    RBVars.S.Qᶠ = Q
  else var == "H"
    RBVars.S.Hₙ = Vecₙ
    RBVars.S.Qʰ = Q
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

function get_Q(RBInfo::Info, RBVars::PoissonSteady)
  if RBVars.Qᵃ == 0
    RBVars.Qᵃ = size(RBVars.Aₙ)[end]
  end
  if !RBInfo.build_Parametric_RHS
    if RBVars.Qᶠ == 0
      RBVars.Qᶠ = size(RBVars.Fₙ)[end]
    end
    if RBVars.Qʰ == 0
      RBVars.Qʰ = size(RBVars.Hₙ)[end]
    end
  end
end

function build_Param_RHS(RBInfo::Info, RBVars::PoissonSGRB, Param, ::Array) ::Tuple
  F = assemble_forcing(FEMSpace, RBInfo, Param)
  H = assemble_neumann_datum(FEMSpace, RBInfo, Param)
  reshape((RBVars.Φₛᵘ)'*F,:,1), reshape((RBVars.Φₛᵘ)'*H,:,1)
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
