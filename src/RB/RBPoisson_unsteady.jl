function get_snapshot_matrix(RBInfo::Info, RBVars::PoissonUnsteady)

  @info "Importing the snapshot matrix for field u, number of snapshots considered: $(RBInfo.nₛ)"

  Sᵘ = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, 1:(RBInfo.nₛ*RBVars.Nₜ)]

  RBVars.S.Sᵘ = Sᵘ
  Nₛᵘ = size(Sᵘ)[1]
  RBVars.S.Nₛᵘ = Nₛᵘ
  RBVars.Nᵘ = RBVars.S.Nₛᵘ * RBVars.Nₜ

  @info "Dimension of the snapshot matrix for field u: $(size(Sᵘ))"

end

function PODs_space(RBInfo::Info, RBVars::PoissonUnsteady)

  @info "Performing the nested spatial POD for field u, using a tolerance of $(RBInfo.ϵₛ)"

  if RBInfo.perform_nested_POD

    for nₛ = 1:RBInfo.nₛ
      Sᵘₙ = RBVars.S.Sᵘ[:, (nₛ-1)*RBVars.Nₜ+1:nₛ*RBVars.Nₜ]
      Φₙ, _ = POD(Sᵘₙ, RBInfo.ϵₛ)
      if nₛ ==1
        global Φₙᵘ_temp = Φₙ
      else
        global Φₙᵘ_temp = hcat(Φₙᵘ_temp, Φₙ)
      end
    end
    Φₛᵘ, _ = POD(Φₙᵘ_temp, RBInfo.ϵₛ)
    RBVars.S.Φₛᵘ = Φₛᵘ
    RBVars.S.nₛᵘ = size(Φₛᵘ)[2]

  else

    PODs_space(RBInfo, RBVars.S)

  end

end


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

  @info "Building the space-time reduced basis for field u, using a tolerance of ($(RBInfo.ϵₛ),$(RBInfo.ϵₜ))"

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

function index_mapping(i::Int, j::Int, RBVars::PoissonUnsteady) :: Int64

  return convert(Int64, (i-1) * RBVars.nₜᵘ + j)

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
        Π_ij = reshape(Φₛᵘ_normed[:, i_s], :, 1) .* reshape(RBVars.Φₜᵘ[:, i_t], :, 1)'
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

function save_M_DEIM_structures(RBInfo::Info, RBVars::PoissonUnsteady)

  list_M_DEIM = (RBVars.MDEIMᵢ_M, RBVars.MDEIM_idx_M, RBVars.sparse_el_M, RBVars.row_idx_A, RBVars.row_idx_M)
  list_names = ("MDEIMᵢ_M", "MDEIM_idx_M", "sparse_el_M", "row_idx_A", "row_idx_M")
  l_info_vec = [[l_idx,l_val] for (l_idx,l_val) in enumerate(list_M_DEIM) if !all(isempty.(l_val))]

  if !isempty(l_info_vec)
    l_info_mat = reduce(vcat,transpose.(l_info_vec))
    l_idx,l_val = l_info_mat[:,1], transpose.(l_info_mat[:,2])
    for (i₁,i₂) in enumerate(l_idx)
      save_CSV(l_val[i₁], joinpath(RBInfo.paths.ROM_structures_path, list_names[i₂]*".csv"))
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
      RBVars.MDEIMᵢ_M = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MDEIMᵢ_M.csv"))
      RBVars.MDEIM_idx_M = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_idx_M.csv"))[:]
      RBVars.sparse_el_M = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "sparse_el_M.csv"))[:]
      RBVars.row_idx_M = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "row_idx_M.csv"))[:]
      append!(operators, [])
    else
      @info "Failed to import MDEIM offline structures for the mass matrix: must build them"
      append!(operators, ["M"])
    end

  end

  if RBInfo.probl_nl["A"]
    if isfile(joinpath(RBInfo.paths.ROM_structures_path, "row_idx_A.csv"))
      RBVars.row_idx_A = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "row_idx_A.csv"))[:]
    else
      @info "Failed to import MDEIM offline structures for the stiffness matrix: must build them"
      append!(operators, ["A"])
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

function modify_timesθ(
  MDEIM_idx::Vector,
  RBInfo::RBUnsteadyProblem,
  RBVars::PoissonUnsteady)
  times_θ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  _, idx_time = from_vec_to_mat_idx(MDEIM_idx, RBVars.S.Nₛᵘ^2)
  times_θ[unique(sort(idx_time))]
end

function modify_MDEIM_idx(MDEIM_idx::Vector, RBVars::PoissonUnsteady) ::Vector
  idx_space, idx_time = from_vec_to_mat_idx(MDEIM_idx,RBVars.S.Nₛᵘ^2)
  idx_time_new = label_sorted_elems(idx_time)
  return (idx_time_new.-1)*RBVars.S.Nₛᵘ^2+idx_space
end

function get_θᵐ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricSpecificsUnsteady) ::Vector
  times_θ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  if !RBInfo.probl_nl["M"]
    θᵐ = [Param.mₜ(t_θ) for t_θ = times_θ]
  else
    M_μ_sparse = build_sparse_mat(FEMInfo, FESpace, Param, RBVars.sparse_el_M, times_θ; var="M")
    Nₛᵘ = RBVars.S.Nₛᵘ
    θᵐ = zeros(RBVars.Qᵐ, RBVars.Nₜ)
    for iₜ = 1:RBVars.Nₜ
      θᵐ[:,iₜ] = M_DEIM_online(M_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ], RBVars.MDEIMᵢ_M, RBVars.MDEIM_idx_M)
    end
  end

  θᵐ = reshape(θᵐ, RBVars.Qᵐ, RBVars.Nₜ)

  return θᵐ

end

function get_θᵐₛₜ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricSpecificsUnsteady) ::Vector

  if !RBInfo.probl_nl["M"]
    θᵐ = [1]
  else
    timesθ_new = modify_timesθ(RBVars.MDEIM_idx_M,RBInfo,RBVars)
    M_μ_sparse = build_sparse_mat(FEMInfo, FESpace, Param, RBVars.sparse_el_M, timesθ_new; var="M")
    MDEIM_idx_new = modify_MDEIM_idx(RBVars.MDEIM_idx_M,RBVars)
    θᵐ = RBVars.MDEIMᵢ_M\Vector(M_μ_sparse[MDEIM_idx_new])
  end

  return θᵐ

end

function get_θᵃ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricSpecificsUnsteady) ::Vector

  if !RBInfo.probl_nl["A"]
    times_θ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
    θᵃ = [Param.αₜ(t_θ,Param.μ) for t_θ = times_θ]
  else
    A_μ_sparse = build_sparse_mat(FEMInfo, FESpace, RBInfo, Param, RBVars.S.sparse_el_A; var="A")
    Nₛᵘ = RBVars.S.Nₛᵘ
    θᵃ = zeros(RBVars.S.Qᵃ, RBVars.Nₜ)
    for iₜ = 1:RBVars.Nₜ
      θᵃ[:,iₜ] = M_DEIM_online(A_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ], RBVars.S.MDEIMᵢ_A, RBVars.S.MDEIM_idx_A)
    end
  end

  θᵃ = reshape(θᵃ, RBVars.S.Qᵃ, RBVars.Nₜ)

  return θᵃ

end

function get_θᵃₛₜ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricSpecificsUnsteady) ::Vector

  if !RBInfo.probl_nl["A"]
    θᵃ = [1]
  else
    timesθ_new = modify_timesθ(RBVars.MDEIM_idx_M,RBInfo,RBVars)
    A_μ_sparse = build_sparse_mat(FEMInfo, FESpace, Param, RBVars.S.sparse_el_A, timesθ_new; var="A")
    MDEIM_idx_new = modify_MDEIM_idx(RBVars.S.MDEIM_idx_A,RBVars)
    θᵃ = RBVars.S.MDEIMᵢ_A\Vector(A_μ_sparse[MDEIM_idx_new])
  end

  return θᵃ

end

function get_θᶠʰ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricSpecificsUnsteady) ::Tuple

  if RBInfo.build_Parametric_RHS
    @error "Cannot fetch θᶠ, θʰ if the RHS is built online"
  end

  times_θ = collect(RBInfo.t₀:RBInfo.δt:RBInfo.T-RBInfo.δt).+RBInfo.δt*RBInfo.θ
  θᶠ, θʰ = Float64[], Float64[]

  if !RBInfo.probl_nl["f"]
    θᶠ = [Param.fₜ(t_θ) for t_θ = times_θ]
  else
    F_μ = assemble_forcing(FESpace, RBInfo, Param)
    for iₜ = 1:RBVars.Nₜ
      append!(θᶠ, M_DEIM_online(F_μ(times_θ[iₜ]), RBVars.S.DEIMᵢ_mat_F, RBVars.S.DEIM_idx_F))
    end
  end

  if !RBInfo.probl_nl["h"]
    θʰ = [Param.hₜ(t_θ) for t_θ = times_θ]
  else
    H_μ = assemble_neumann_datum(FESpace, RBInfo, Param)
    for iₜ = 1:RBVars.Nₜ
      append!(θʰ, M_DEIM_online(H_μ(times_θ[iₜ]), RBVars.S.DEIMᵢ_mat_H, RBVars.S.DEIM_idx_H))
    end
  end

  θᶠ = reshape(θᶠ, RBVars.S.Qᶠ, RBVars.Nₜ)
  θʰ = reshape(θʰ, RBVars.S.Qʰ, RBVars.Nₜ)

  return θᶠ, θʰ

end

function get_θᶠʰₛₜ(RBInfo::Info, RBVars::RBUnsteadyProblem, Param::ParametricSpecificsUnsteady) ::Tuple

  if RBInfo.build_Parametric_RHS
    @error "Cannot fetch θᶠ, θʰ if the RHS is built online"
  end

  if !RBInfo.probl_nl["f"]
    θᶠ = [1]
  else
    F_μ = assemble_forcing(FESpace, RBInfo, Param)
    DEIM_idx_new = modify_MDEIM_idx(RBVars.S.DEIM_idx_F,RBVars)
    θᶠ = RBVars.S.DEIMᵢ_F\Vector(F_μ[DEIM_idx_new])
  end

  if !RBInfo.probl_nl["h"]
    θʰ = [1]
  else
    H_μ = assemble_neumann_datum(FESpace, RBInfo, Param)
    DEIM_idx_new = modify_MDEIM_idx(RBVars.S.DEIM_idx_H,RBVars)
    θʰ = RBVars.S.DEIMᵢ_H\Vector(H_μ[DEIM_idx_new])
  end

  return θᶠ, θʰ

end

function get_Q(RBInfo::Info, RBVars::PoissonUnsteady)

  if RBVars.Qᵐ == 0
    RBVars.Qᵐ = size(RBVars.Mₙ)[end]
  end

  get_Q(RBInfo, RBVars.S)

end

function solve_RB_system(RBInfo::Info, RBVars::PoissonUnsteady, Param::ParametricSpecificsUnsteady)

  get_RB_system(RBInfo, RBVars, Param)

  @info "Solving RB problem via backslash"
  @info "Condition number of the system's matrix: $(cond(RBVars.S.LHSₙ[1]))"
  RBVars.S.uₙ = zeros(RBVars.nᵘ)
  RBVars.S.uₙ = RBVars.S.LHSₙ[1] \ RBVars.S.RHSₙ[1]

end

function reconstruct_FEM_solution(RBVars::PoissonUnsteady)

  @info "Reconstructing FEM solution from the newly computed RB one"

  uₙ = reshape(RBVars.S.uₙ, (RBVars.nₜᵘ, RBVars.S.nₛᵘ))
  RBVars.S.ũ = RBVars.S.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'

end

function offline_phase(RBInfo::Info, RBVars::PoissonUnsteady)

  RBVars.Nₜ = convert(Int64, RBInfo.T / RBInfo.δt)

  @info "Building $(RBInfo.RB_method) approximation with $(RBInfo.nₛ) snapshots and tolerances of $(RBInfo.ϵₛ) in space"

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
    @error "Impossible to assemble the reduced problem if neither the snapshots nor the bases can be loaded"
  end

  if import_snapshots_success && !import_basis_success
    @info "Failed to import the reduced basis, building it via POD"
    build_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.import_offline_structures
    operators = get_offline_structures(RBInfo, RBVars)
    if "A" ∈ operators || "M" ∈ operators || "MA" ∈ operators || "F" ∈ operators
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

end

function online_phase(RBInfo::Info, RBVars::PoissonUnsteady, μ, Param_nbs)

  H1_L2_err = zeros(length(Param_nbs))
  mean_H1_err = zeros(RBVars.Nₜ)
  mean_H1_L2_err = 0.0
  mean_pointwise_err = zeros(RBVars.S.Nₛᵘ, RBVars.Nₜ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(RBInfo, RBVars.S)

  ũ_μ = zeros(RBVars.S.Nₛᵘ, length(Param_nbs)*RBVars.Nₜ)
  uₙ_μ = zeros(RBVars.nᵘ, length(Param_nbs))

  for (i_nb, nb) in enumerate(Param_nbs)
    println("\n")
    @info "Considering Parameter number: $nb"

    μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
    Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ_nb)
    uₕ_test = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, (nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]

    online_time = @elapsed begin
      solve_RB_system(RBInfo, RBVars, Param)
    end
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    mean_online_time = online_time / length(Param_nbs)
    mean_reconstruction_time = reconstruction_time / length(Param_nbs)

    H1_err_nb, H1_L2_err_nb = compute_errors(uₕ_test, RBVars, RBVars.S.Xᵘ₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(Param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(Param_nbs)
    mean_pointwise_err += abs.(uₕ_test - RBVars.S.ũ) / length(Param_nbs)

    ũ_μ[:, (i_nb-1)*RBVars.Nₜ+1:i_nb*RBVars.Nₜ] = RBVars.S.ũ
    uₙ_μ[:, i_nb] = RBVars.S.uₙ

    @info "Online wall time: $online_time s (snapshot number $nb)"
    @info "Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)"

  end

  string_Param_nbs = "Params"
  for Param_nb in Param_nbs
    string_Param_nbs *= "_" * string(Param_nb)
  end
  path_μ = joinpath(RBInfo.paths.results_path, string_Param_nbs)

  if RBInfo.save_results

    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV(mean_H1_err, joinpath(path_μ, "H1_err.csv"))
    save_CSV([mean_H1_L2_err], joinpath(path_μ, "H1L2_err.csv"))

    if !RBInfo.import_offline_structures
      times = [RBVars.S.offline_time, mean_online_time, mean_reconstruction_time]
    else
      times = [mean_online_time, mean_reconstruction_time]
    end
    save_CSV(times, joinpath(path_μ, "times.csv"))

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FESpace"=>FESpace, "H1_L2_err"=>H1_L2_err, "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>mean_pointwise_err)

  if RBInfo.postprocess
    post_process(RBInfo, pass_to_pp)
  end

  #=
  plot_stability_constant(FESpace,RBInfo,Param,Nₜ)
  =#

end

function post_process(RBInfo::UnsteadyInfo, d::Dict)

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    generate_and_save_plot(MDEIM_Σ, "Decay singular values, MDEIM", "σ index", "σ value", RBInfo.paths.results_path)
  end
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    generate_and_save_plot(DEIM_Σ, "Decay singular values, DEIM", "σ index", "σ value", RBInfo.paths.results_path)
  end

  times = collect(RBInfo.t₀+RBInfo.δt:RBInfo.δt:RBInfo.T)
  FESpace = d["FESpace"]
  vtk_dir = joinpath(d["path_μ"], "vtk_folder")

  create_dir(vtk_dir)
  createpvd(joinpath(vtk_dir,"mean_point_err_u")) do pvd
    for (i,t) in enumerate(times)
      errₕt = FEFunction(FESpace.V(t), d["mean_point_err_u"][:,i])
      pvd[i] = createvtk(FESpace.Ω, joinpath(vtk_dir, "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
    end
  end

  generate_and_save_plot(d["mean_H1_err"], "Average ||uₕ(t) - ũ(t)||ₕ₁", "time [s]", "H¹ error", d["path_μ"])
  generate_and_save_plot(d["H1_L2_err"], "||uₕ - ũ||ₕ₁₋ₗ₂", "Param μ number", "H¹-L² error", d["path_μ"])

  if length(keys(d)) == 8

    createpvd(joinpath(vtk_dir,"mean_point_err_p")) do pvd
      for (i,t) in enumerate(times)
        errₕt = FEFunction(FESpace.Q, d["mean_point_err_p"][:,i])
        pvd[i] = createvtk(FESpace.Ω, joinpath(vtk_dir, "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
      end
    end

    generate_and_save_plot(d["mean_L2_err"], "Average ||pₕ(t) - p̃(t)||ₗ₂", "time [s]", "L² error", d["path_μ"])
    generate_and_save_plot(d["L2_L2_err"], "||pₕ - p̃||ₗ₂₋ₗ₂", "Param μ number", "L²-L² error", d["path_μ"])

  end

end

function check_dataset(
  FESpace::FEMProblem,
  RBInfo::Info,
  RBVars::PoissonUnsteady,
  nb::Int64)

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  μ_i = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
  Param = get_Parametric_specifics(problem_ntuple, RBInfo, μ_i)

  u1 = RBVars.S.Sᵘ[:,(nb-1)*RBVars.Nₜ+1]
  u2 = RBVars.S.Sᵘ[:,(nb-1)*RBVars.Nₜ+2]
  M = assemble_mass(FESpace, RBInfo, Param)(0.0)
  # we suppose that case == 1 --> no need to multiply A by αₜ
  A(t) = assemble_stiffness(FESpace, RBInfo, Param)(t)
  F = assemble_forcing(FESpace, RBInfo, Param)(0.0)
  H = assemble_neumann_datum(FESpace, RBInfo, Param)(0.0)

  t¹_θ = RBInfo.t₀+RBInfo.δt*RBInfo.θ
  t²_θ = t¹_θ+RBInfo.δt

  LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t¹_θ))
  RHS1 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t¹_θ)+H*Param.hₜ(t¹_θ))
  my_u1 = LHS1\RHS1

  LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))
  mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A(t²_θ))-M
  RHS2 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  my_u2 = LHS2\RHS2

  u1≈my_u1 && u2≈my_u2

end

function plot_stability_constants(
  FESpace::FEMProblem,
  RBInfo::Info,
  Param::ParametricSpecificsUnsteady)

  M = assemble_mass(FESpace, RBInfo, Param)(0.0)
  A = assemble_stiffness(FESpace, RBInfo, Param)(0.0)
  stability_constants = []
  for Nₜ = 10:10:1000
    const_Nₜ = compute_stability_constant(RBInfo,Nₜ,M,A)
    append!(stability_constants, const_Nₜ)
  end
  pyplot()
  p = plot(collect(10:10:1000), stability_constants, xaxis=:log, yaxis=:log, lw = 3, label="||(Aˢᵗ)⁻¹||₂", title = "Euclidean norm of (Aˢᵗ)⁻¹", legend=:topleft)
  p = plot!(collect(10:10:1000), collect(10:10:1000), xaxis=:log, yaxis=:log, lw = 3, label="Nₜ")
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
