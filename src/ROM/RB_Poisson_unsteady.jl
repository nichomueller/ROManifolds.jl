include("RB_Poisson_steady.jl")
include("ST-GRB_Poisson.jl")
include("ST-PGRB_Poisson.jl")

function get_snapshot_matrix(ROM_info::Info, RB_variables::PoissonUnsteady)

  @info "Importing the snapshot matrix for field u, number of snapshots considered: $(ROM_info.nₛ)"

  Sᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, 1:(ROM_info.nₛ*RB_variables.Nₜ)]

  RB_variables.S.Sᵘ = Sᵘ
  Nₛᵘ = size(Sᵘ)[1]
  RB_variables.S.Nₛᵘ = Nₛᵘ
  RB_variables.Nᵘ = RB_variables.S.Nₛᵘ * RB_variables.Nₜ

  @info "Dimension of the snapshot matrix for field u: $(size(Sᵘ))"

end

function PODs_space(ROM_info::Info, RB_variables::PoissonUnsteady)

  @info "Performing the nested spatial POD for field u, using a tolerance of $(ROM_info.ϵₛ)"

  if ROM_info.perform_nested_POD

    for nₛ = 1:ROM_info.nₛ
      Sᵘₙ = RB_variables.S.Sᵘ[:, (nₛ-1)*RB_variables.Nₜ+1:nₛ*RB_variables.Nₜ]
      Φₙ, _ = POD(Sᵘₙ, ROM_info.ϵₛ)
      if nₛ ===1
        global Φₙᵘ_temp = Φₙ
      else
        global Φₙᵘ_temp = hcat(Φₙᵘ_temp, Φₙ)
      end
    end
    Φₛᵘ, _ = POD(Φₙᵘ_temp, ROM_info.ϵₛ)
    RB_variables.S.Φₛᵘ = Φₛᵘ
    RB_variables.S.nₛᵘ = size(Φₛᵘ)[2]

  else

    PODs_space(ROM_info, RB_variables.S)

  end

end


function PODs_time(ROM_info::Info, RB_variables::PoissonUnsteady)

  @info "Performing the temporal POD for field u, using a tolerance of $(ROM_info.ϵₜ)"

  if ROM_info.time_reduction_technique === "ST-HOSVD"
    Sᵘₜ = zeros(RB_variables.Nₜ, RB_variables.S.nₛᵘ * ROM_info.nₛ)
    Sᵘ = RB_variables.S.Φₛᵘ' * RB_variables.S.Sᵘ
    for i in 1:ROM_info.nₛ
      Sᵘₜ[:, (i-1)*RB_variables.S.nₛᵘ+1:i*RB_variables.S.nₛᵘ] =
      Sᵘ[:, (i-1)*RB_variables.Nₜ+1:i*RB_variables.Nₜ]'
    end
  else
    Sᵘₜ = zeros(RB_variables.Nₜ, RB_variables.S.Nₛᵘ * ROM_info.nₛ)
    Sᵘ = RB_variables.S.Sᵘ
    for i in 1:ROM_info.nₛ
      Sᵘₜ[:, (i-1)*RB_variables.S.Nₛᵘ+1:i*RB_variables.S.Nₛᵘ] =
      transpose(Sᵘ[:, (i-1)*RB_variables.Nₜ+1:i*RB_variables.Nₜ])
    end
  end

  Φₜᵘ, _ = POD(Sᵘₜ, ROM_info.ϵₜ)
  RB_variables.Φₜᵘ = Φₜᵘ
  RB_variables.nₜᵘ = size(Φₜᵘ)[2]

end

function build_reduced_basis(ROM_info::Info, RB_variables::PoissonUnsteady)

  @info "Building the space-time reduced basis for field u, using a tolerance of ($(ROM_info.ϵₛ),$(ROM_info.ϵₜ))"

  RB_building_time = @elapsed begin
    PODs_space(ROM_info, RB_variables)
    PODs_time(ROM_info, RB_variables)
  end

  RB_variables.nᵘ = RB_variables.S.nₛᵘ * RB_variables.nₜᵘ
  RB_variables.Nᵘ = RB_variables.S.Nₛᵘ * RB_variables.Nₜ

  RB_variables.S.offline_time += RB_building_time

  if ROM_info.save_offline_structures
    save_CSV(RB_variables.S.Φₛᵘ, joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
    save_CSV(RB_variables.Φₜᵘ, joinpath(ROM_info.paths.basis_path, "Φₜᵘ.csv"))
  end

end

function import_reduced_basis(ROM_info::Info, RB_variables::PoissonUnsteady)

  import_reduced_basis(ROM_info, RB_variables.S)

  @info "Importing the temporal reduced basis for field u"
  RB_variables.Φₜᵘ = load_CSV(joinpath(ROM_info.paths.basis_path, "Φₜᵘ.csv"))
  RB_variables.nₜᵘ = size(RB_variables.Φₜᵘ)[2]
  RB_variables.nᵘ = RB_variables.S.nₛᵘ * RB_variables.nₜᵘ

end

function index_mapping(i::Int, j::Int, RB_variables::PoissonUnsteady) :: Int64

  return convert(Int64, (i-1) * RB_variables.nₜᵘ + j)

end

function get_generalized_coordinates(ROM_info::Info, RB_variables::PoissonUnsteady, snaps=nothing)

  if check_norm_matrix(RB_variables.S)
    get_norm_matrix(ROM_info, RB_variables.S)
  end

  if snaps === nothing || maximum(snaps) > ROM_info.nₛ
    snaps = 1:ROM_info.nₛ
  end

  û = zeros(RB_variables.nᵘ, length(snaps))
  Φₛᵘ_normed = RB_variables.S.Xᵘ₀ * RB_variables.S.Φₛᵘ

  for (i, i_nₛ) = enumerate(snaps)
    @info "Assembling generalized coordinate relative to snapshot $(i_nₛ), field u"
    S_i = RB_variables.S.Sᵘ[:, (i_nₛ-1)*RB_variables.Nₜ+1:i_nₛ*RB_variables.Nₜ]
    for i_s = 1:RB_variables.S.nₛᵘ
      for i_t = 1:RB_variables.nₜᵘ
        Π_ij = reshape(Φₛᵘ_normed[:, i_s], :, 1) .* reshape(RB_variables.Φₜᵘ[:, i_t], :, 1)'
        û[index_mapping(i_s, i_t, RB_variables), i] = sum(Π_ij .* S_i)
      end
    end
  end

  RB_variables.S.û = û

  if ROM_info.save_offline_structures
    save_CSV(û, joinpath(ROM_info.paths.gen_coords_path, "û.csv"))
  end

end

function test_offline_phase(ROM_info::Info, RB_variables::PoissonUnsteady)

  get_generalized_coordinates(ROM_info, RB_variables, 1)

  uₙ = reshape(RB_variables.S.û, (RB_variables.nₜᵘ, RB_variables.S.nₛᵘ))
  u_rec = RB_variables.S.Φₛᵘ * (RB_variables.Φₜᵘ * uₙ)'
  err = zeros(RB_variables.Nₜ)
  for i = 1:RB_variables.Nₜ
    err[i] = compute_errors(RB_variables.S.Sᵘ[:, i], u_rec[:, i])
  end

end

function save_M_DEIM_structures(ROM_info::Info, RB_variables::PoissonUnsteady)

  list_M_DEIM = (RB_variables.MDEIMᵢ_M, RB_variables.MDEIM_idx_M, RB_variables.sparse_el_M, RB_variables.row_idx_A, RB_variables.row_idx_M)
  list_names = ("MDEIMᵢ_M", "MDEIM_idx_M", "sparse_el_M", "row_idx_A", "row_idx_M")
  l_info_vec = [[l_idx,l_val] for (l_idx,l_val) in enumerate(list_M_DEIM) if !all(isempty.(l_val))]

  if !isempty(l_info_vec)
    l_info_mat = reduce(vcat,transpose.(l_info_vec))
    l_val = l_info_mat[:,2]
    for (i,v) in enumerate(l_val)
      save_CSV(v,joinpath(ROM_info.paths.ROM_structures_path, list_names[i]*".csv"))
    end
  end

  save_M_DEIM_structures(ROM_info, RB_variables.S)

end

function set_operators(ROM_info, RB_variables::PoissonUnsteady) :: Vector

  return vcat(["M"], set_operators(ROM_info, RB_variables.S))

end

function get_ST_M_DEIM_structures(ROM_info::Info, RB_variables::PoissonUnsteady) :: Vector

  operators = String[]

  if ROM_info.probl_nl["A"]
    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "row_idx_A.csv"))
      RB_variables.row_idx_A = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "row_idx_A.csv"))[:]
    else
      @info "Failed to import MDEIM offline structures for the stiffness matrix, space-time technique: must build them"
      append!(operators, ["A"])
    end
  end
  if ROM_info.probl_nl["M"]
    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "row_idx_M.csv"))
      RB_variables.row_idx_M = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "row_idx_M.csv"))[:]
    else
      @info "Failed to import MDEIM offline structures for the mass matrix, space-time technique: must build them"
      append!(operators, ["M"])
    end
  end

  operators

end

function get_M_DEIM_structures(ROM_info::Info, RB_variables::PoissonUnsteady) :: Vector

  operators = String[]

  if ROM_info.probl_nl["M"]

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "MDEIMᵢ_M.csv"))
      @info "Importing MDEIM offline structures for the mass matrix"
      RB_variables.MDEIMᵢ_M = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIMᵢ_M.csv"))
      RB_variables.MDEIM_idx_M = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_idx_M.csv"))[:]
      RB_variables.sparse_el_M = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "sparse_el_M.csv"))[:]
      return []
    else
      @info "Failed to import MDEIM offline structures for the mass matrix: must build them"
      append!(operators, ["M"])
    end

  end

  append!(operators, get_M_DEIM_structures(ROM_info, RB_variables.S))
  if ROM_info.space_time_M_DEIM
    append!(operators, get_ST_M_DEIM_structures(ROM_info, RB_variables))
  end

end

function get_offline_structures(ROM_info::Info, RB_variables::PoissonUnsteady) :: Vector

  operators = String[]
  append!(operators, get_affine_structures(ROM_info, RB_variables))
  append!(operators, get_M_DEIM_structures(ROM_info, RB_variables))
  unique!(operators)

  operators

end

function get_θᵐ(ROM_info::Info, RB_variables::RBUnsteadyProblem, param::ParametricSpecificsUnsteady) :: Array

  if !ROM_info.probl_nl["M"]
    times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
    θᵐ = [param.mₜ(t_θ) for t_θ = times_θ]
  else
    M_μ_sparse = build_sparse_mat(problem_info, FE_space, ROM_info, param, RB_variables.sparse_el_M; var="M")
    Nₛᵘ = RB_variables.S.Nₛᵘ
    θᵐ = zeros(RB_variables.Qᵐ, RB_variables.Nₜ)
    for iₜ = 1:RB_variables.Nₜ
      θᵐ[:,iₜ] = M_DEIM_online(M_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ], RB_variables.MDEIMᵢ_M, RB_variables.MDEIM_idx_M)
    end
  end

  θᵐ = reshape(θᵐ, RB_variables.Qᵐ, RB_variables.Nₜ)

  return θᵐ

end

function get_θᵐₛₜ(ROM_info::Info, RB_variables::RBUnsteadyProblem, param::ParametricSpecificsUnsteady) :: Array

  if !ROM_info.probl_nl["M"]
    θᵐ = get_θᵐ(ROM_info, RB_variables, param)
  else
    Nₛᵘ = RB_variables.S.Nₛᵘ
    _, MDEIM_idx_time = from_spacetime_to_space_time_idx_mat(RB_variables.MDEIM_idx_M, Nₛᵘ)
    unique!(MDEIM_idx_time)
    M_μ_sparse = build_sparse_mat(problem_info, FE_space, ROM_info, param, RB_variables.sparse_el_M; var="M")

    θᵐ = zeros(RB_variables.Qᵐ, length(MDEIM_idx_time))
    for iₜ = 1:length(MDEIM_idx_time)
      θᵐ[:,iₜ] = M_DEIM_online(M_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ], RB_variables.MDEIMᵢ_M, RB_variables.MDEIM_idx_M)
    end
  end

  return θᵐ

end

function get_θᵃ(ROM_info::Info, RB_variables::RBUnsteadyProblem, param::ParametricSpecificsUnsteady) :: Array

  if !ROM_info.probl_nl["A"]
    times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
    θᵃ = [param.αₜ(t_θ,param.μ) for t_θ = times_θ]
  else
    A_μ_sparse = build_sparse_mat(problem_info, FE_space, ROM_info, param, RB_variables.S.sparse_el_A; var="A")
    Nₛᵘ = RB_variables.S.Nₛᵘ
    θᵃ = zeros(RB_variables.S.Qᵃ, RB_variables.Nₜ)
    for iₜ = 1:RB_variables.Nₜ
      θᵃ[:,iₜ] = M_DEIM_online(A_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ], RB_variables.S.MDEIMᵢ_A, RB_variables.S.MDEIM_idx_A)
    end
  end

  θᵃ = reshape(θᵃ, RB_variables.S.Qᵃ, RB_variables.Nₜ)

  return θᵃ

end

function get_θᵃₛₜ(ROM_info::Info, RB_variables::RBUnsteadyProblem, param::ParametricSpecificsUnsteady) :: Array

  if !ROM_info.probl_nl["A"]
    θᵃ = get_θᵃ(ROM_info, RB_variables, param)
  else
    A_μ_sparse = build_sparse_mat(problem_info, FE_space, ROM_info, param, RB_variables.S.sparse_el_A, RB_variables.S.MDEIM_idx_A; var="A")
    MDEIM_idx_new = modify_MDEIM_idx(RB_variables.S.MDEIM_idx_A,RB_variables.S.Nₛᵘ^2)
    θᵃ = MDEIMᵢ_mat\Vector(A_μ_sparse[MDEIM_idx_new])
  end

  return θᵃ

end

function get_θᶠʰ(ROM_info::Info, RB_variables::RBUnsteadyProblem, param::ParametricSpecificsUnsteady) :: Tuple

  if ROM_info.build_parametric_RHS
    @error "Cannot fetch θᶠ, θʰ if the RHS is built online"
  end

  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  θᶠ, θʰ = Float64[], Float64[]

  if !ROM_info.probl_nl["f"]
    θᶠ = [param.fₜ(t_θ) for t_θ = times_θ]
  else
    F_μ, _ = assemble_forcing(FE_space, ROM_info, param)
    for iₜ = 1:RB_variables.Nₜ
      append!(θᶠ, M_DEIM_online(F_μ(times_θ[iₜ]), RB_variables.S.DEIMᵢ_mat_F, RB_variables.S.DEIM_idx_F))
    end
  end

  if !ROM_info.probl_nl["h"]
    θʰ = [param.hₜ(t_θ) for t_θ = times_θ]
  else
    _, H_μ = assemble_forcing(FE_space, ROM_info, param)
    for iₜ = 1:RB_variables.Nₜ
      append!(θʰ, M_DEIM_online(H_μ(times_θ[iₜ]), RB_variables.S.DEIMᵢ_mat_H, RB_variables.S.DEIM_idx_H))
    end
  end

  θᶠ = reshape(θᶠ, RB_variables.S.Qᶠ, RB_variables.Nₜ)
  θʰ = reshape(θʰ, RB_variables.S.Qʰ, RB_variables.Nₜ)

  return θᶠ, θʰ

end

function get_θᶠʰₛₜ(ROM_info::Info, RB_variables::RBUnsteadyProblem, param::ParametricSpecificsUnsteady) :: Tuple

  if ROM_info.build_parametric_RHS
    @error "Cannot fetch θᶠ, θʰ if the RHS is built online"
  end

  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  θᶠ, θʰ = Float64[], Float64[]

  if !ROM_info.probl_nl["f"]
    θᶠ = [param.fₜ(t_θ) for t_θ = times_θ]
  else
    F_μ, _ = assemble_forcing(FE_space, ROM_info, param)
    _, DEIM_idx_time = from_spacetime_to_space_time_idx_vec(RB_variables.S.DEIM_idx_F, RB_variables.S.Nₛᵘ)
    unique!(DEIM_idx_time)
    times_DEIM = times_θ[DEIM_idx_time]
    for tᵢ = times_DEIM
      append!(θᶠ, M_DEIM_online(F_μ(tᵢ), RB_variables.S.DEIMᵢ_mat_F, RB_variables.S.DEIM_idx_F))
    end
  end

  if !ROM_info.probl_nl["h"]
    θʰ = [param.hₜ(t_θ) for t_θ = times_θ]
  else
    _, H_μ = assemble_forcing(FE_space, ROM_info, param)
    _, DEIM_idx_time = from_spacetime_to_space_time_idx_vec(RB_variables.S.DEIM_idx_H, RB_variables.S.Nₛᵘ)
    unique!(DEIM_idx_time)
    times_DEIM = times_θ[DEIM_idx_time]
    for tᵢ = times_DEIM
      append!(θʰ, M_DEIM_online(H_μ(tᵢ), RB_variables.S.DEIMᵢ_mat_H, RB_variables.S.DEIM_idx_H))
    end
  end

  θᶠ = reshape(θᶠ, RB_variables.S.Qᶠ, RB_variables.Nₜ)
  θʰ = reshape(θʰ, RB_variables.S.Qʰ, RB_variables.Nₜ)

  return θᶠ, θʰ

end

function get_Q(ROM_info::Info, RB_variables::PoissonUnsteady)

  if RB_variables.Qᵐ === 0
    RB_variables.Qᵐ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Qᵐ.csv"))[1]
  end

  get_Q(ROM_info, RB_variables.S)

end

function solve_RB_system(ROM_info::Info, RB_variables::PoissonUnsteady, param::ParametricSpecificsUnsteady)

  get_RB_system(ROM_info, RB_variables, param)

  @info "Solving RB problem via backslash"
  @info "Condition number of the system's matrix: $(cond(RB_variables.S.LHSₙ[1]))"
  RB_variables.S.uₙ = zeros(RB_variables.nᵘ)
  RB_variables.S.uₙ = RB_variables.S.LHSₙ[1] \ RB_variables.S.RHSₙ[1]

end

function reconstruct_FEM_solution(RB_variables::PoissonUnsteady)

  @info "Reconstructing FEM solution from the newly computed RB one"

  uₙ = reshape(RB_variables.S.uₙ, (RB_variables.nₜᵘ, RB_variables.S.nₛᵘ))
  RB_variables.S.ũ = RB_variables.S.Φₛᵘ * (RB_variables.Φₜᵘ * uₙ)'

end

function build_RB_approximation(ROM_info::Info, RB_variables::PoissonUnsteady)

  RB_variables.Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)

  @info "Building $(ROM_info.RB_method) approximation with $(ROM_info.nₛ) snapshots and tolerances of $(ROM_info.ϵₛ) in space"

  if ROM_info.import_snapshots
    get_snapshot_matrix(ROM_info, RB_variables)
    import_snapshots_success = true
  else
    import_snapshots_success = false
  end

  if ROM_info.import_offline_structures
    import_reduced_basis(ROM_info, RB_variables)
    import_basis_success = true
  else
    import_basis_success = false
  end

  if !import_snapshots_success && !import_basis_success
    @error "Impossible to assemble the reduced problem if neither the snapshots nor the bases can be loaded"
  end

  if import_snapshots_success && !import_basis_success
    @info "Failed to import the reduced basis, building it via POD"
    build_reduced_basis(ROM_info, RB_variables)
  end

  if ROM_info.import_offline_structures
    operators = get_offline_structures(ROM_info, RB_variables)
    if "A" ∈ operators || "M" ∈ operators || "MA" ∈ operators || "F" ∈ operators
      assemble_offline_structures(ROM_info, RB_variables, operators)
    end
  else
    assemble_offline_structures(ROM_info, RB_variables)
  end

end

function testing_phase(ROM_info::Info, RB_variables::PoissonUnsteady, μ, param_nbs)

  H1_L2_err = zeros(length(param_nbs))
  mean_H1_err = zeros(RB_variables.Nₜ)
  mean_H1_L2_err = 0.0
  mean_pointwise_err = zeros(RB_variables.S.Nₛᵘ, RB_variables.Nₜ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(ROM_info, RB_variables.S)

  ũ_μ = zeros(RB_variables.S.Nₛᵘ, length(param_nbs)*RB_variables.Nₜ)
  uₙ_μ = zeros(RB_variables.nᵘ, length(param_nbs))

  for (i_nb, nb) in enumerate(param_nbs)
    @info "Considering parameter number: $nb"

    μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_nb)
    uₕ_test = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, (nb-1)*RB_variables.Nₜ+1:nb*RB_variables.Nₜ]

    online_time = @elapsed begin
      solve_RB_system(ROM_info, RB_variables, parametric_info)
    end
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RB_variables)
    end
    mean_online_time = online_time / length(param_nbs)
    mean_reconstruction_time = reconstruction_time / length(param_nbs)

    H1_err_nb, H1_L2_err_nb = compute_errors(uₕ_test, RB_variables, RB_variables.S.Xᵘ₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err += abs.(uₕ_test - RB_variables.S.ũ) / length(param_nbs)

    ũ_μ[:, (i_nb-1)*RB_variables.Nₜ+1:i_nb*RB_variables.Nₜ] = RB_variables.S.ũ
    uₙ_μ[:, i_nb] = RB_variables.S.uₙ

    @info "Online wall time: $online_time s (snapshot number $nb)"
    @info "Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)"

  end

  string_param_nbs = "params"
  for param_nb in param_nbs
    string_param_nbs *= "_" * string(param_nb)
  end
  path_μ = joinpath(ROM_info.paths.results_path, string_param_nbs)

  if ROM_info.save_results

    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV(mean_H1_err, joinpath(path_μ, "H1_err.csv"))
    save_CSV([mean_H1_L2_err], joinpath(path_μ, "H1L2_err.csv"))

    if !ROM_info.import_offline_structures
      times = [RB_variables.S.offline_time, mean_online_time, mean_reconstruction_time]
    else
      times = [mean_online_time, mean_reconstruction_time]
    end
    save_CSV(times, joinpath(path_μ, "times.csv"))

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FE_space"=>FE_space, "H1_L2_err"=>H1_L2_err, "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>mean_pointwise_err)

  if ROM_info.postprocess
    post_process(ROM_info, pass_to_pp)
  end

  #= stability_constants = []
  for Nₜ = 10:10:1000
    append!(stability_constants, compute_stability_constant(ROM_info, M, A, ROM_info.θ, Nₜ))
  end
  pyplot()
  p = plot(collect(10:10:1000), stability_constants, xaxis=:log, yaxis=:log, lw = 3, label="||(Aˢᵗ)⁻¹||₂", title = "Euclidean norm of (Aˢᵗ)⁻¹", legend=:topleft)
  p = plot!(collect(10:10:1000), collect(10:10:1000), xaxis=:log, yaxis=:log, lw = 3, label="Nₜ")
  xlabel!("Nₜ")
  savefig(p, joinpath(ROM_info.paths.results_path, "stability_constant.eps"))
  =#

end

function check_dataset(ROM_info, RB_variables, i)

  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  μ_i = parse.(Float64, split(chop(μ[i]; head=1, tail=1), ','))
  param = get_parametric_specifics(ROM_info, μ_i)

  u1 = RB_variables.S.Sᵘ[:, (i-1)*RB_variables.Nₜ+1]
  u2 = RB_variables.S.Sᵘ[:, (i-1)*RB_variables.Nₜ+2]
  M = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "M.csv"); convert_to_sparse = true)
  A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
  F = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "F.csv"))
  H = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "H.csv"))

  t¹_θ = ROM_info.t₀+ROM_info.δt*ROM_info.θ
  t²_θ = t¹_θ+ROM_info.δt

  LHS1 = ROM_info.θ*(M+ROM_info.δt*ROM_info.θ*A*param.αₜ(t¹_θ,μ_i))
  RHS1 = ROM_info.δt*ROM_info.θ*(F*param.fₜ(t¹_θ)+H*param.hₜ(t¹_θ))
  my_u1 = LHS1\RHS1

  LHS2 = ROM_info.θ*(M+ROM_info.δt*ROM_info.θ*A*param.αₜ(t²_θ,μ_i))
  mat = (1-ROM_info.θ)*(M+ROM_info.δt*ROM_info.θ*A*param.αₜ(t²_θ,μ_i))-M
  RHS2 = ROM_info.δt*ROM_info.θ*(F*param.fₜ(t²_θ)+H*param.hₜ(t²_θ))-mat*u1
  my_u2 = LHS2\RHS2

  u1≈my_u1
  u2≈my_u2

end

function compute_stability_constant(ROM_info, M, A, θ, Nₜ)

  #= M = assemble_mass(FE_space, ROM_info, parametric_info)(0.0)
  A = assemble_stiffness(FE_space, ROM_info, parametric_info)(0.0) =#
  Nₕ = size(M)[1]
  δt = ROM_info.T/Nₜ
  B₁ = θ*(M + θ*δt*A)
  B₂ = θ*(-M + (1-θ)*δt*A)
  λ₁,_ = eigs(B₁)
  λ₂,_ = eigs(B₂)

  return 1/(minimum(abs.(λ₁)) + minimum(abs.(λ₂)))

end
