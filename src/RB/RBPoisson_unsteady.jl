include("RBPoisson_steady.jl")
include("ST-GRB_Poisson.jl")
include("ST-PGRB_Poisson.jl")

function get_snapshot_matrix(RBInfo::Info, RBVars::PoissonUnsteady)

  if RBInfo.perform_nested_POD
    @info "Importing the snapshot matrix for field u obtained
      with the nested POD"
    Sᵘ = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"),
      DataFrame))
  else
    @info "Importing the snapshot matrix for field u,
      number of snapshots considered: $(RBInfo.nₛ)"
    Sᵘ = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"),
      DataFrame))[:, 1:(RBInfo.nₛ*RBVars.Nₜ)]
  end

  RBVars.S.Sᵘ = Sᵘ
  Nₛᵘ = size(Sᵘ)[1]
  RBVars.S.Nₛᵘ = Nₛᵘ
  RBVars.Nᵘ = RBVars.S.Nₛᵘ * RBVars.Nₜ

  @info "Dimension of the snapshot matrix for field u: $(size(Sᵘ))"

end

PODs_space(RBInfo::Info, RBVars::PoissonUnsteady) = PODs_space(RBInfo, RBVars.S)

function PODs_time(RBInfo::Info, RBVars::PoissonUnsteady)

  @info "Performing the temporal POD for field u, using a tolerance of $(RBInfo.ϵₜ)"

  if RBInfo.time_reduction_technique == "ST-HOSVD"
    Sᵘₜ = zeros(RBVars.Nₜ, RBVars.S.nₛᵘ * RBInfo.nₛ)
    Sᵘ = RBVars.S.Φₛᵘ' * RBVars.S.Sᵘ
    for i in 1:RBInfo.nₛ
      Sᵘₜ[:, (i-1)*RBVars.S.nₛᵘ+1:i*RBVars.S.nₛᵘ] =
      Sᵘ[:, (i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ]'
    end
  else
    Sᵘₜ = zeros(RBVars.Nₜ, RBVars.S.Nₛᵘ * RBInfo.nₛ)
    Sᵘ = RBVars.S.Sᵘ
    for i in 1:RBInfo.nₛ
      Sᵘₜ[:, (i-1)*RBVars.S.Nₛᵘ+1:i*RBVars.S.Nₛᵘ] =
      transpose(Sᵘ[:, (i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ])
    end
  end

  Φₜᵘ, _ = POD(Sᵘₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵘ = Φₜᵘ
  RBVars.nₜᵘ = size(Φₜᵘ)[2]

end

function build_reduced_basis(RBInfo::Info, RBVars::PoissonUnsteady)

  @info "Building the space-time reduced basis for field u"

  RB_building_time = @elapsed begin
    PODs_space(RBInfo, RBVars)
    PODs_time(RBInfo, RBVars)
  end

  RBVars.nᵘ = RBVars.S.nₛᵘ * RBVars.nₜᵘ
  RBVars.Nᵘ = RBVars.S.Nₛᵘ * RBVars.Nₜ

  RBVars.S.offline_time += RB_building_time

  if RBInfo.save_offline_structures
    save_CSV(RBVars.S.Φₛᵘ, joinpath(RBInfo.paths.basis_path, "Φₛᵘ.csv"))
    save_CSV(RBVars.Φₜᵘ, joinpath(RBInfo.paths.basis_path, "Φₜᵘ.csv"))
  end

end

function import_reduced_basis(RBInfo::Info, RBVars::PoissonUnsteady)

  import_reduced_basis(RBInfo, RBVars.S)

  @info "Importing the temporal reduced basis for field u"
  RBVars.Φₜᵘ = load_CSV(joinpath(RBInfo.paths.basis_path, "Φₜᵘ.csv"))
  RBVars.nₜᵘ = size(RBVars.Φₜᵘ)[2]
  RBVars.nᵘ = RBVars.S.nₛᵘ * RBVars.nₜᵘ

end

function index_mapping(i::Int,j::Int,RBVars::PoissonUnsteady) ::Int
  Int((i-1)*RBVars.nₜᵘ+j)
end

function index_mapping_inverse(i::Int,RBVars::PoissonUnsteady) ::Tuple
  iₛ = 1+floor(Int64,(i-1)/RBVars.nₜᵘ)
  iₜ = i-(iₛ-1)*RBVars.nₜᵘ
  iₛ,iₜ
end

function get_generalized_coordinates(RBInfo::Info, RBVars::PoissonUnsteady, snaps=nothing)

  if check_norm_matrix(RBVars.S)
    get_norm_matrix(RBInfo, RBVars.S)
  end

  if isnothing(snaps) || maximum(snaps) > RBInfo.nₛ
    snaps = 1:RBInfo.nₛ
  end

  û = zeros(RBVars.nᵘ, length(snaps))
  Φₛᵘ_normed = RBVars.S.Xᵘ₀ * RBVars.S.Φₛᵘ

  for (i, i_nₛ) = enumerate(snaps)
    @info "Assembling generalized coordinate relative to snapshot $(i_nₛ), field u"
    S_i = RBVars.S.Sᵘ[:, (i_nₛ-1)*RBVars.Nₜ+1:i_nₛ*RBVars.Nₜ]
    for i_s = 1:RBVars.S.nₛᵘ
      for i_t = 1:RBVars.nₜᵘ
        Π_ij = reshape(Φₛᵘ_normed[:,i_s],:,1).*reshape(RBVars.Φₜᵘ[:,i_t],:,1)'
        û[index_mapping(i_s, i_t, RBVars), i] = sum(Π_ij .* S_i)
      end
    end
  end

  RBVars.S.û = û

  if RBInfo.save_offline_structures
    save_CSV(û, joinpath(RBInfo.paths.gen_coords_path, "û.csv"))
  end

end

function test_offline_phase(RBInfo::Info, RBVars::PoissonUnsteady)

  get_generalized_coordinates(RBInfo, RBVars, 1)

  uₙ = reshape(RBVars.S.û, (RBVars.nₜᵘ, RBVars.S.nₛᵘ))
  u_rec = RBVars.S.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'
  err = zeros(RBVars.Nₜ)
  for i = 1:RBVars.Nₜ
    err[i] = compute_errors(RBVars.S.Sᵘ[:, i], u_rec[:, i])
  end

end

function assemble_MDEIM_matrices(
  RBInfo::Info,
  RBVars::PoissonUnsteady,
  var::String)

  if var == "M"
    @info "The matrix $var is non-affine:
      running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots"
    if isempty(RBVars.MDEIM_mat_M)
      (RBVars.MDEIM_mat_M, RBVars.MDEIM_idx_M, RBVars.MDEIMᵢ_M, RBVars.row_idx_M,
        RBVars.sparse_el_M) = MDEIM_offline(FEMSpace, RBInfo, "M")
    end
    assemble_reduced_mat_MDEIM(
      RBInfo,RBVars,RBVars.MDEIM_mat_M,RBVars.row_idx_M,"M")
  elseif var == "A"
    if isempty(RBVars.S.MDEIM_mat_A)
      (RBVars.S.MDEIM_mat_A, RBVars.S.MDEIM_idx_A, RBVars.S.MDEIMᵢ_A,
      RBVars.S.row_idx_A,RBVars.S.sparse_el_A) = MDEIM_offline(FEMSpace, RBInfo, "A")
    end
    assemble_reduced_mat_MDEIM(
      RBInfo,RBVars,RBVars.S.MDEIM_mat_A,RBVars.S.row_idx_A,"A")
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_DEIM_vectors(
  RBInfo::Info,
  RBVars::PoissonUnsteady,
  var::String)

  @info "The vector $var is non-affine:
    running the DEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots"

  if var == "F"
    if isempty(RBVars.S.DEIM_mat_F)
       RBVars.S.DEIM_mat_F, RBVars.S.DEIM_idx_F, RBVars.S.DEIMᵢ_F =
        DEIM_offline(FEMSpace,RBInfo,"F")
    end
    assemble_reduced_mat_DEIM(RBInfo,RBVars,RBVars.S.DEIM_mat_F,"F")
  elseif var == "H"
    if isempty(RBVars.S.DEIM_mat_H)
       RBVars.S.DEIM_mat_H, RBVars.S.DEIM_idx_H, RBVars.S.DEIMᵢ_H =
        DEIM_offline(FEMSpace,RBInfo,"H")
    end
    assemble_reduced_mat_DEIM(RBInfo,RBVars, RBVars.S.DEIM_mat_H,"H")
  else
    error("Unrecognized variable on which to perform DEIM")
  end

end

function save_M_DEIM_structures(RBInfo::Info, RBVars::PoissonUnsteady)

  list_M_DEIM = (RBVars.MDEIM_mat_M, RBVars.MDEIMᵢ_M, RBVars.MDEIM_idx_M,
    RBVars.sparse_el_M, RBVars.row_idx_M)
  list_names = ("MDEIM_mat_M", "MDEIMᵢ_M", "MDEIM_idx_M", "sparse_el_M",
   "row_idx_M")
  l_info_vec = [[l_idx,l_val] for (l_idx,l_val) in
    enumerate(list_M_DEIM) if !all(isempty.(l_val))]

  if !isempty(l_info_vec)
    l_info_mat = reduce(vcat,transpose.(l_info_vec))
    l_idx,l_val = l_info_mat[:,1], transpose.(l_info_mat[:,2])
    for (i₁,i₂) in enumerate(l_idx)
      save_CSV(l_val[i₁], joinpath(RBInfo.paths.ROM_structures_path,
        list_names[i₂]*".csv"))
    end
  end

  save_M_DEIM_structures(RBInfo, RBVars.S)

end

function set_operators(RBInfo, RBVars::PoissonUnsteady) :: Vector

  return vcat(["M"], set_operators(RBInfo, RBVars.S))

end

function get_M_DEIM_structures(RBInfo::Info, RBVars::PoissonUnsteady) :: Vector

  operators = String[]

  if RBInfo.probl_nl["M"]

    if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MDEIMᵢ_M.csv"))
      @info "Importing MDEIM offline structures for the mass matrix"
      RBVars.MDEIMᵢ_M = load_CSV(joinpath(RBInfo.paths.ROM_structures_path,
        "MDEIMᵢ_M.csv"))
      RBVars.MDEIM_idx_M = load_CSV(joinpath(RBInfo.paths.ROM_structures_path,
        "MDEIM_idx_M.csv"))[:]
      RBVars.sparse_el_M = load_CSV(joinpath(RBInfo.paths.ROM_structures_path,
        "sparse_el_M.csv"))[:]
      RBVars.row_idx_M = load_CSV(joinpath(RBInfo.paths.ROM_structures_path,
        "row_idx_M.csv"))[:]
      append!(operators, [])
    else
      @info "Failed to import MDEIM offline structures for the mass matrix: must build them"
      append!(operators, ["M"])
    end

  end

  append!(operators, get_M_DEIM_structures(RBInfo, RBVars.S))

end

function get_offline_structures(RBInfo::Info, RBVars::PoissonUnsteady) ::Vector

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_θᵐ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricInfoUnsteady) ::Array
  timesθ = get_timesθ(RBInfo)
  if !RBInfo.probl_nl["M"]
    θᵐ = [Param.mₜ(t_θ) for t_θ = timesθ]
  else
    M_μ_sparse = build_sparse_mat(
      FEMInfo,FEMSpace,Param,RBVars.S.sparse_el_A,timesθ;var="M")
    Nₛᵘ = RBVars.S.Nₛᵘ
    θᵐ = zeros(RBVars.Qᵐ, RBVars.Nₜ)
    for iₜ = 1:RBVars.Nₜ
      θᵐ[:,iₜ] = M_DEIM_online(M_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ],
        RBVars.MDEIMᵢ_M, RBVars.MDEIM_idx_M)
    end
  end

  θᵐ = reshape(θᵐ, RBVars.Qᵐ, RBVars.Nₜ)

  return θᵐ

end

function get_θᵐₛₜ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricInfoUnsteady) ::Array

  if !RBInfo.probl_nl["M"]
    θᵐ = [1]
  else
    timesθ_mod,MDEIM_idx_mod =
      modify_timesθ_and_MDEIM_idx(RBVars.MDEIM_idx_M,RBInfo,RBVars)
    M_μ_sparse = build_sparse_mat(FEMInfo, FEMSpace, Param,
      RBVars.sparse_el_M, timesθ_mod; var="M")
    θᵐ = RBVars.MDEIMᵢ_M\Vector(M_μ_sparse[MDEIM_idx_mod])
  end

  return θᵐ

end

function get_θᵃ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricInfoUnsteady) ::Array

  timesθ = get_timesθ(RBInfo)
  if !RBInfo.probl_nl["A"]
    θᵃ = [Param.αₜ(t_θ,Param.μ) for t_θ = timesθ]
  else
    A_μ_sparse = build_sparse_mat(
      FEMInfo,FEMSpace,Param,RBVars.S.sparse_el_A,timesθ;var="A")
    Nₛᵘ = RBVars.S.Nₛᵘ
    θᵃ = zeros(RBVars.S.Qᵃ, RBVars.Nₜ)
    for iₜ = 1:RBVars.Nₜ
      θᵃ[:,iₜ] = M_DEIM_online(A_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ],
        RBVars.S.MDEIMᵢ_A, RBVars.S.MDEIM_idx_A)
    end
  end

  θᵃ = reshape(θᵃ, RBVars.S.Qᵃ, RBVars.Nₜ)

  return θᵃ

end

function get_θᵃₛₜ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricInfoUnsteady) ::Array

  if !RBInfo.probl_nl["A"]
    θᵃ = [1]
  else
    timesθ_mod,MDEIM_idx_mod =
      modify_timesθ_and_MDEIM_idx(RBVars.S.MDEIM_idx_A,RBInfo,RBVars)
    A_μ_sparse = build_sparse_mat(FEMInfo, FEMSpace, Param,
      RBVars.S.sparse_el_A, timesθ_mod; var="A")
    θᵃ = RBVars.S.MDEIMᵢ_A\Vector(A_μ_sparse[MDEIM_idx_mod])
  end

  return θᵃ

end

function get_θᶠʰ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricInfoUnsteady) ::Tuple

  if RBInfo.build_Parametric_RHS
    error("Cannot fetch θᶠ, θʰ if the RHS is built online")
  end

  timesθ = get_timesθ(RBInfo)
  θᶠ, θʰ = Float64[], Float64[]

  if !RBInfo.probl_nl["f"]
    θᶠ = [Param.fₜ(t_θ) for t_θ = timesθ]
  else
    F_μ = assemble_forcing(FEMSpace, RBInfo, Param)
    for iₜ = 1:RBVars.Nₜ
      append!(θᶠ,
        M_DEIM_online(F_μ(timesθ[iₜ]), RBVars.S.DEIMᵢ_F, RBVars.S.DEIM_idx_F))
    end
  end

  if !RBInfo.probl_nl["h"]
    θʰ = [Param.hₜ(t_θ) for t_θ = timesθ]
  else
    H_μ = assemble_neumann_datum(FEMSpace, RBInfo, Param)
    for iₜ = 1:RBVars.Nₜ
      append!(θʰ,
        M_DEIM_online(H_μ(timesθ[iₜ]), RBVars.S.DEIMᵢ_H, RBVars.S.DEIM_idx_H))
    end
  end

  θᶠ = reshape(θᶠ, RBVars.S.Qᶠ, RBVars.Nₜ)
  θʰ = reshape(θʰ, RBVars.S.Qʰ, RBVars.Nₜ)

  return θᶠ, θʰ

end

function get_θᶠʰₛₜ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricInfoUnsteady) ::Tuple

  if RBInfo.build_Parametric_RHS
    error("Cannot fetch θᶠ, θʰ if the RHS is built online")
  end

  if !RBInfo.probl_nl["f"]
    θᶠ = [1]
  else
    F_μ = assemble_forcing(FEMSpace, RBInfo, Param)
    _,DEIM_idx_mod = modify_timesθ_and_MDEIM_idx(RBVars.S.DEIM_idx_F,RBInfo,RBVars)
    θᶠ = RBVars.S.DEIMᵢ_F\Vector(F_μ[DEIM_idx_mod])
  end

  if !RBInfo.probl_nl["h"]
    θʰ = [1]
  else
    H_μ = assemble_neumann_datum(FEMSpace, RBInfo, Param)
    _,DEIM_idx_mod = modify_timesθ_and_MDEIM_idx(RBVars.S.DEIM_idx_H,RBInfo,RBVars)
    θʰ = RBVars.S.DEIMᵢ_H\Vector(H_μ[DEIM_idx_mod])
  end

  return θᶠ, θʰ

end

function solve_RB_system(RBInfo::Info, RBVars::PoissonUnsteady, Param::ParametricInfoUnsteady)
  get_RB_system(RBInfo, RBVars, Param)
  @info "Solving RB problem via backslash"
  @info "Condition number of the system's matrix: $(cond(RBVars.S.LHSₙ[1]))"
  RBVars.S.online_time += @elapsed begin
    RBVars.S.uₙ = zeros(RBVars.nᵘ)
    RBVars.S.uₙ = RBVars.S.LHSₙ[1] \ RBVars.S.RHSₙ[1]
  end
end

function reconstruct_FEM_solution(RBVars::PoissonUnsteady)
  @info "Reconstructing FEM solution from the newly computed RB one"
  uₙ = reshape(RBVars.S.uₙ, (RBVars.nₜᵘ, RBVars.S.nₛᵘ))
  RBVars.S.ũ = RBVars.S.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'
end

function offline_phase(RBInfo::Info, RBVars::PoissonUnsteady)

  RBVars.Nₜ = convert(Int64, RBInfo.T / RBInfo.δt)

  if RBInfo.import_snapshots
    get_snapshot_matrix(RBInfo, RBVars)
    import_snapshots_success = true
  else
    import_snapshots_success = false
  end

  if RBInfo.import_offline_structures
    import_reduced_basis(RBInfo, RBVars)
    import_basis_success = true
  else
    import_basis_success = false
  end

  if !import_snapshots_success && !import_basis_success
    error("Impossible to assemble the reduced problem if neither
      the snapshots nor the bases can be loaded")
  end

  if import_snapshots_success && !import_basis_success
    @info "Failed to import the reduced basis, building it via POD"
    build_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.import_offline_structures
    operators = get_offline_structures(RBInfo, RBVars)
    if !isempty(operators)
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

end

function loop_on_params(
  RBInfo::Info,
  RBVars::PoissonUnsteady,
  μ::Matrix,
  param_nbs) ::Tuple

  H1_L2_err = zeros(length(param_nbs))
  mean_H1_err = zeros(RBVars.Nₜ)
  mean_H1_L2_err = 0.0
  mean_pointwise_err = zeros(RBVars.S.Nₛᵘ, RBVars.Nₜ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  ũ_μ = zeros(RBVars.S.Nₛᵘ, length(param_nbs)*RBVars.Nₜ)
  uₙ_μ = zeros(RBVars.nᵘ, length(param_nbs))

  for (i_nb, nb) in enumerate(param_nbs)
    println("\n")
    @info "Considering Parameter number: $nb/$(param_nbs[end])"

    μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μ_nb)
    if RBInfo.perform_nested_POD
      nb_test = nb-90
      uₕ_test = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path,
      "uₕ_test.csv"), DataFrame))[:,(nb_test-1)*RBVars.Nₜ+1:nb_test*RBVars.Nₜ]
    else
      uₕ_test = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"),
      DataFrame))[:,(nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]
    end

    solve_RB_system(RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    if i_nb > 1
      mean_online_time = online_time/(length(param_nbs)-1)
      mean_reconstruction_time = reconstruction_time/(length(param_nbs)-1)
    end

    H1_err_nb, H1_L2_err_nb = compute_errors(uₕ_test, RBVars, RBVars.S.Xᵘ₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err += abs.(uₕ_test - RBVars.S.ũ) / length(param_nbs)

    ũ_μ[:, (i_nb-1)*RBVars.Nₜ+1:i_nb*RBVars.Nₜ] = RBVars.S.ũ
    uₙ_μ[:, i_nb] = RBVars.S.uₙ

    @info "Online wall time: $online_time s (snapshot number $nb)"
    @info "Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)"
  end
  return (ũ_μ,uₙ_μ,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,H1_L2_err,
  mean_online_time,mean_reconstruction_time)
end

function online_phase(
  RBInfo::Info,
  RBVars::PoissonUnsteady,
  μ::Matrix,
  param_nbs)

  get_norm_matrix(RBInfo, RBVars.S)
  (ũ_μ,uₙ_μ,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,H1_L2_err,
    mean_online_time,mean_reconstruction_time) =
    loop_on_params(RBInfo, RBVars, μ, param_nbs)

  adaptive_loop = false
  if adaptive_loop
    (ũ_μ,uₙ_μ,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,H1_L2_err,
      mean_online_time,mean_reconstruction_time) =
      adaptive_loop_on_params(RBInfo,RBVars,mean_pointwise_err)
  end

  string_param_nbs = "Params"
  for Param_nb in param_nbs
    string_param_nbs *= "_" * string(Param_nb)
  end
  path_μ = joinpath(RBInfo.paths.results_path, string_param_nbs)

  if RBInfo.save_results
    @info "Saving the results..."
    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV(mean_H1_err, joinpath(path_μ, "H1_err.csv"))
    save_CSV([mean_H1_L2_err], joinpath(path_μ, "H1L2_err.csv"))

    if !RBInfo.import_offline_structures
      times = Dict(RBVars.S.offline_time=>"off_time",
        mean_online_time=>"on_time", mean_reconstruction_time=>"rec_time")
    else
      times = Dict(mean_online_time=>"on_time",
        mean_reconstruction_time=>"rec_time")
    end
    CSV.write(joinpath(path_μ, "times.csv"),times)
  end

  pass_to_pp = Dict("path_μ"=>path_μ,
    "FEMSpace"=>FEMSpace, "H1_L2_err"=>H1_L2_err,
    "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>mean_pointwise_err)

  if RBInfo.postprocess
    @info "Post-processing the results..."
    post_process(RBInfo, pass_to_pp)
  end

  #=
  plot_stability_constant(FEMSpace,RBInfo,Param,Nₜ)
  =#

end

function post_process(RBInfo::UnsteadyInfo, d::Dict)
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(MDEIM_Σ), MDEIM_Σ, "Decay singular values, MDEIM",
      ["σ"], "σ index", "σ value", RBInfo.paths.results_path; var="MDEIM_Σ")
  end
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(DEIM_Σ), DEIM_Σ, "Decay singular values, DEIM",
      ["σ"], "σ index", "σ value", RBInfo.paths.results_path; var="DEIM_Σ")
  end

  times = collect(RBInfo.t₀+RBInfo.δt:RBInfo.δt:RBInfo.T)
  FEMSpace = d["FEMSpace"]
  vtk_dir = joinpath(d["path_μ"], "vtk_folder")

  create_dir(vtk_dir)
  createpvd(joinpath(vtk_dir,"mean_point_err_u")) do pvd
    for (i,t) in enumerate(times)
      errₕt = FEFunction(FEMSpace.V(t), d["mean_point_err_u"][:,i])
      pvd[i] = createvtk(FEMSpace.Ω, joinpath(vtk_dir,
        "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
    end
  end

  generate_and_save_plot(times,d["mean_H1_err"],
    "Average ||uₕ(t) - ũ(t)||ₕ₁", ["H¹ err"], "time [s]", "H¹ error", d["path_μ"];
    var="H1_err")
  xvec = collect(eachindex(d["H1_L2_err"]))
  generate_and_save_plot(xvec,d["H1_L2_err"],
    "||uₕ - ũ||ₕ₁₋ₗ₂", ["H¹-l² err"], "Param μ number", "H¹-l² error", d["path_μ"];
    var="H1_L2_err")

  if length(keys(d)) == 8

    createpvd(joinpath(vtk_dir,"mean_point_err_p")) do pvd
      for (i,t) in enumerate(times)
        errₕt = FEFunction(FEMSpace.Q, d["mean_point_err_p"][:,i])
        pvd[i] = createvtk(FEMSpace.Ω, joinpath(vtk_dir,
          "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
      end
    end

    generate_and_save_plot(times,d["mean_L2_err"],
      "Average ||pₕ(t) - p̃(t)||ₗ₂", ["l² err"], "time [s]", "L² error", d["path_μ"];
      var="L2_err")
    xvec = collect(eachindex(d["L2_L2_err"]))
    generate_and_save_plot(xvec,d["L2_L2_err"],
      "||pₕ - p̃||ₗ₂₋ₗ₂", ["l²-l² err"], "Param μ number", "L²-L² error", d["path_μ"];
      var="L2_L2_err")

  end

end

function plot_stability_constants(
  FEMSpace::FEMProblem,
  RBInfo::Info,
  Param::ParametricInfoUnsteady)

  M = assemble_mass(FEMSpace, RBInfo, Param)(0.0)
  A = assemble_stiffness(FEMSpace, RBInfo, Param)(0.0)
  stability_constants = []
  for Nₜ = 10:10:1000
    const_Nₜ = compute_stability_constant(RBInfo,Nₜ,M,A)
    append!(stability_constants, const_Nₜ)
  end
  p = Plot.plot(collect(10:10:1000),
    stability_constants, xaxis=:log, yaxis=:log, lw = 3,
    label="||(Aˢᵗ)⁻¹||₂", title = "Euclidean norm of (Aˢᵗ)⁻¹", legend=:topleft)
  p = Plot.plot!(collect(10:10:1000), collect(10:10:1000),
    xaxis=:log, yaxis=:log, lw = 3, label="Nₜ")
  xlabel!("Nₜ")
  savefig(p, joinpath(RBInfo.paths.results_path, "stability_constant.eps"))

  function compute_stability_constant(RBInfo,Nₜ,M,A)
    δt = RBInfo.T/Nₜ
    B₁ = RBInfo.θ*(M + RBInfo.θ*δt*A)
    B₂ = RBInfo.θ*(-M + (1-RBInfo.θ)*δt*A)
    λ₁,_ = eigs(B₁)
    λ₂,_ = eigs(B₂)
    return 1/(minimum(abs.(λ₁)) + minimum(abs.(λ₂)))
  end

end
