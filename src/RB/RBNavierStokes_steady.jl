include("RBStokes_steady.jl")
include("S-GRB_NavierStokes.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady{T}) where T

  get_snapshot_matrix(RBInfo, RBVars.Stokes)

end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::NavierStokesSteady{T}) where T

  get_norm_matrix(RBInfo, RBVars.Stokes)

end

function build_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

  build_reduced_basis(RBInfo, RBVars.Stokes)

end

function import_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady{T}) where T

  import_reduced_basis(RBInfo, RBVars.Stokes)

end

function set_operators(
  RBInfo::Info,
  RBVars::NavierStokesSteady)

  set_operators(RBInfo, RBVars.Stokes)

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  var::String)

  assemble_MDEIM_matrices(RBInfo, RBVars.Stokes, var)

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.Stokes, var)

end

function save_M_DEIM_structures(
  ::ROMInfoSteady,
  ::NavierStokesSteady)

  error("not implemented")

end

function get_M_DEIM_structures(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

  get_M_DEIM_structures(RBInfo, RBVars.Stokes)

end

function get_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

  operators = String[]

  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_system_blocks(
  RBInfo::Info,
  RBVars::NavierStokesSteady,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int})

  get_system_blocks(RBInfo, RBVars.Stokes, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::NavierStokesSteady,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  save_system_blocks(RBInfo, RBVars.Stokes, LHS_blocks, RHS_blocks, operators)

end

function get_θᵃ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  Param::ParametricInfoSteady)

  get_θᵃ(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function get_θᵇ(
  ::SteadyProblem,
  ::ROMInfoSteady{T},
  ::NavierStokesSteady,
  ::ParametricInfoSteady) where T

  reshape([one(T)],1,1)::Matrix{T}

end

function get_θᶠʰ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  Param::ParametricInfoSteady)

  get_θᶠʰ(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function initialize_RB_system(RBVars::NavierStokesSteady)
  initialize_RB_system(RBVars.Stokes)
end

function initialize_online_time(RBVars::NavierStokesSteady)
  initialize_RB_system(RBVars.Stokes)
end

function get_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  Param::ParametricInfoSteady)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    LHS_blocks = [1, 2, 3]
    RHS_blocks = [1]
    operators = get_system_blocks(RBInfo, RBVars, LHS_blocks, RHS_blocks)

    θᵃ, θᶠ, θʰ, θᵇ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      println("Assembling reduced LHS")
      push!(RBVars.LHSₙ, get_RB_LHS_blocks(RBInfo, RBVars, θᵃ, θᵇ))
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        println("Assembling reduced RHS")
        push!(RBVars.RHSₙ, get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ))
      else
        println("Assembling reduced RHS exactly")
        build_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,LHS_blocks,RHS_blocks,operators)

end

function solve_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::NavierStokesSteady,
  Param::ParametricInfoSteady) where T

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)
  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")
  RBVars.online_time += @elapsed begin
    xₙ = (vcat(hcat(RBVars.LHSₙ[1], RBVars.LHSₙ[2]),
      hcat(RBVars.LHSₙ[3], zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ))) \
      vcat(RBVars.RHSₙ[1], zeros(T, RBVars.nₛᵖ, 1)))
  end

  RBVars.uₙ = xₙ[1:RBVars.nₛᵘ,:]
  RBVars.pₙ = xₙ[RBVars.nₛᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::NavierStokesSteady)
  println("Reconstructing FEM solution from the newly computed RB one")
  reconstruct_FEM_solution(RBVars.Stokes)
  RBVars.p̃ = RBVars.Φₛᵖ * RBVars.pₙ
end

function offline_phase(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

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
    error("Impossible to assemble the reduced problem if
      neither the snapshots nor the bases can be loaded")
  end

  if import_snapshots_success && !import_basis_success
    println("Failed to import the reduced basis, building it via POD")
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

function online_phase(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady{T},
  param_nbs) where T

  μ = load_CSV(Array{T}[],
    joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)

  mean_H1_err = 0.0
  mean_L2_err = 0.0
  mean_pointwise_err_u = zeros(T, RBVars.Nₛᵘ)
  mean_pointwise_err_p = zeros(T, RBVars.Nₛᵖ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(RBInfo, RBVars)

  ũ_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs))
  uₙ_μ = zeros(T, RBVars.nₛᵘ, length(param_nbs))
  p̃_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs))
  pₙ_μ = zeros(T, RBVars.nₛᵘ, length(param_nbs))

  for nb in param_nbs
    println("Considering parameter number: $nb")

    Param = get_ParamInfo(RBInfo, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:, nb]
    pₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
      DataFrame))[:, nb]

    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    mean_online_time = RBVars.online_time / length(param_nbs)
    mean_reconstruction_time = reconstruction_time / length(param_nbs)

    H1_err_nb = compute_errors(uₕ_test, RBVars, RBVars.Xᵘ₀)
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test - RBVars.ũ) / length(param_nbs)

    L2_err_nb = compute_errors(pₕ_test, RBVars, RBVars.Xᵖ₀)
    mean_L2_err += L2_err_nb / length(param_nbs)
    mean_pointwise_err_p += abs.(pₕ_test - RBVars.p̃) / length(param_nbs)

    ũ_μ[:, nb - param_nbs[1] + 1] = RBVars.ũ
    uₙ_μ[:, nb - param_nbs[1] + 1] = RBVars.uₙ
    p̃_μ[:, nb - param_nbs[1] + 1] = RBVars.p̃
    pₙ_μ[:, nb - param_nbs[1] + 1] = RBVars.pₙ

    println("Online wall time: $(RBVars.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-error: $H1_err_nb (snapshot number $nb)")
    println("Relative reconstruction L2-error: $L2_err_nb (snapshot number $nb)")

  end

  string_param_nbs = "params"
  for Param_nb in param_nbs
    string_param_nbs *= "_" * string(Param_nb)
  end
  path_μ = joinpath(RBInfo.Paths.results_path, string_param_nbs)

  if RBInfo.save_results

    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err_u, joinpath(path_μ, "mean_point_err_u.csv"))
    save_CSV([mean_H1_err], joinpath(path_μ, "H1_err.csv"))
    save_CSV(p̃_μ, joinpath(path_μ, "p̃.csv"))
    save_CSV(pₙ_μ, joinpath(path_μ, "pₙ.csv"))
    save_CSV(mean_pointwise_err_p, joinpath(path_μ, "mean_point_err_p.csv"))
    save_CSV([mean_L2_err], joinpath(path_μ, "L2_err.csv"))

    if RBInfo.import_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time,"rec_time"=>mean_reconstruction_time)

    CSV.write(joinpath(path_μ, "times.csv"),times)

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FEMSpace"=>FEMSpace,
    "mean_point_err_u"=>Float.(mean_pointwise_err_u),
    "mean_point_err_p"=>Float.(mean_pointwise_err_p))

  if RBInfo.post_process
    post_process(RBInfo, pass_to_pp)
  end

end
