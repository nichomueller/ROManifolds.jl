include("RBPoisson_steady.jl")
include("S-GRB_Stokes.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady{T}) where T

  get_snapshot_matrix(RBInfo, RBVars.P)

  println("Importing the snapshot matrix for field u,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵖ = Matrix{T}(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "pₕ.csv"),
    DataFrame))[:, 1:RBInfo.nₛ]
  println("Dimension of pressure snapshot matrix: $(size(Sᵖ))")
  RBVars.Sᵖ = Sᵖ
  RBVars.Nₛᵖ = size(Sᵖ)[1]

end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::StokesSteady{T}) where T

  get_norm_matrix(RBInfo, RBVars.P)

  if check_norm_matrix(RBVars)

    println("Importing the norm matrices Xᵘ, Xᵖ₀")

    Xᵘ = load_CSV(sparse([],[],T[]),
      joinpath(RBInfo.paths.FEM_structures_path, "Xᵘ.csv"))
    Xᵖ₀ = load_CSV(sparse([],[],T[]),
      joinpath(RBInfo.paths.FEM_structures_path, "Xᵖ₀.csv"))
    RBVars.Nₛᵖ = size(Xᵖ₀)[1]
    println("Dimension of L² norm matrix, field p: $(size(Xᵖ₀))")

    if RBInfo.use_norm_X
      RBVars.Xᵘ = Xᵘ
      RBVars.Xᵖ₀ = Xᵖ₀
    else
      RBVars.Xᵘ = one(T)*sparse(I,RBVars.P.Nₛᵘ,RBVars.P.Nₛᵘ)
      RBVars.Xᵖ₀ = one(T)*sparse(I,RBVars.Nₛᵖ,RBVars.Nₛᵖ)
    end

  end

end

function check_norm_matrix(RBVars::StokesSteady)
  isempty(RBVars.Xᵘ) || isempty(RBVars.Xᵖ₀)
end

function PODs_space(
  RBInfo::Info,
  RBVars::StokesSteady)

  PODs_space(RBInfo,RBVars.P)

  println("Performing the spatial POD for field p, using a tolerance of $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars)
  RBVars.Φₛᵖ, _ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.Xᵖ₀)
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function primal_supremizers(
  RBInfo::Info,
  RBVars::StokesSteady{T}) where T

  println("Computing primal supremizers")

  #dir_idx = abs.(diag(RBVars.Xᵘ) .- 1) .< 1e-16

  constraint_mat = load_CSV(sparse([],[],T[]),
    joinpath(RBInfo.paths.FEM_structures_path, "B.csv"))'
  #constraint_mat[dir_idx[dir_idx≤RBVars.P.Nₛᵘ*RBVars.Nₛᵖ]] = 0

  supr_primal = Matrix{T}(RBVars.Xᵘ) \ (Matrix{T}(constraint_mat) * RBVars.Φₛᵖ)

  min_norm = 1e16
  for i = 1:size(supr_primal)[2]

    println("Normalizing primal supremizer $i")

    for j in 1:RBVars.P.nₛᵘ
      supr_primal[:, i] -= mydot(supr_primal[:, i], RBVars.P.Φₛᵘ[:,j], RBVars.P.Xᵘ₀) /
      mynorm(RBVars.P.Φₛᵘ[:,j], RBVars.P.Xᵘ₀) * RBVars.P.Φₛᵘ[:,j]
    end
    for j in 1:i
      supr_primal[:, i] -= mydot(supr_primal[:, i], supr_primal[:, j], RBVars.P.Xᵘ₀) /
      mynorm(supr_primal[:, j], RBVars.P.Xᵘ₀) * supr_primal[:, j]
    end

    supr_norm = mynorm(supr_primal[:, i], RBVars.P.Xᵘ₀)
    min_norm = min(supr_norm, min_norm)
    println("Norm supremizers: $supr_norm")
    supr_primal[:, i] /= supr_norm

  end

  println("Primal supremizers enrichment ended with norm: $min_norm")

  supr_primal

end

function supr_enrichment_space(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady)

  supr_primal = primal_supremizers(RBInfo, RBVars)
  RBVars.P.Φₛᵘ = hcat(RBVars.P.Φₛᵘ, supr_primal)
  RBVars.P.nₛᵘ = size(RBVars.P.Φₛᵘ)[2]

end

function build_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady)

  RBVars.P.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
    supr_enrichment_space(RBInfo, RBVars)
  end

  if RBInfo.save_offline_structures
    save_CSV(RBVars.P.Φₛᵘ, joinpath(RBInfo.paths.basis_path,"Φₛᵘ.csv"))
    save_CSV(RBVars.Φₛᵖ, joinpath(RBInfo.paths.basis_path,"Φₛᵖ.csv"))
  end

end

function import_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady{T}) where T

  import_reduced_basis(RBInfo, RBVars.P)

  println("Importing the spatial reduced basis for field p")
  RBVars.Φₛᵖ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.paths.basis_path, "Φₛᵖ.csv"))
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function get_generalized_coordinates(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  snaps=nothing)

  get_norm_matrix(RBInfo, RBVars)
  if isnothing(snaps) || maximum(snaps) > RBInfo.nₛ
    snaps = 1:RBInfo.nₛ
  end

  get_generalized_coordinates(RBInfo, RBVars.P, snaps)

  Φₛᵖ_normed = RBVars.Xᵖ₀*RBVars.Φₛᵖ
  RBVars.p̂ = RBVars.Sᵖ[:,snaps]*Φₛᵖ_normed
  if RBInfo.save_offline_structures
    save_CSV(RBVars.p̂, joinpath(RBInfo.paths.gen_coords_path, "p̂.csv"))
  end

end

function set_operators(
  RBInfo::Info,
  RBVars::StokesSteady)

  append!(["B"], set_operators(RBInfo, RBVars.P))

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  var::String)

  assemble_MDEIM_matrices(RBInfo, RBVars.P, var)

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.P, var)

end

function save_M_DEIM_structures(
  ::ROMInfoSteady,
  ::StokesSteady)

  error("not implemented")

end

function get_M_DEIM_structures(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady)

  get_M_DEIM_structures(RBInfo, RBVars.P)

end

function get_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady)

  operators = String[]

  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_system_blocks(
  RBInfo::Info,
  RBVars::StokesSteady,
  LHS_blocks::Vector{Int64},
  RHS_blocks::Vector{Int64})

  get_system_blocks(RBInfo, RBVars.P, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::StokesSteady,
  LHS_blocks::Vector{Int64},
  RHS_blocks::Vector{Int64},
  operators::Vector{String})

  save_system_blocks(RBInfo, RBVars.P, LHS_blocks, RHS_blocks, operators)

end

function get_θᵃ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  Param::ParametricInfoSteady)

  get_θᵃ(FEMSpace, RBInfo, RBVars.P, Param)

end

function get_θᵇ(
  ::SteadyProblem,
  ::ROMInfoSteady{T},
  ::StokesSteady,
  ::ParametricInfoSteady) where T

  reshape([one(T)],1,1)::Matrix{T}

end

function get_θᶠʰ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  Param::ParametricInfoSteady)

  get_θᶠʰ(FEMSpace, RBInfo, RBVars.P, Param)

end

function initialize_RB_system(RBVars::StokesSteady)
  initialize_RB_system(RBVars.P)
end

function initialize_online_time(RBVars::StokesSteady)
  initialize_RB_system(RBVars.P)
end

function get_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  Param::ParametricInfoSteady)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.P.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    LHS_blocks = [1, 2, 3]
    RHS_blocks = [1]
    operators = get_system_blocks(RBInfo, RBVars, LHS_blocks, RHS_blocks)

    θᵃ, θᶠ, θʰ, θᵇ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      println("Assembling reduced LHS")
      push!(RBVars.P.LHSₙ, get_RB_LHS_blocks(RBInfo, RBVars, θᵃ, θᵇ))
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        println("Assembling reduced RHS")
        push!(RBVars.P.RHSₙ, get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ))
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
  RBVars::StokesSteady,
  Param::ParametricInfoSteady) where T

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)
  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.P.LHSₙ[1]))")
  RBVars.P.online_time += @elapsed begin
    xₙ = (vcat(hcat(RBVars.P.LHSₙ[1], RBVars.P.LHSₙ[2]),
      hcat(RBVars.P.LHSₙ[3], zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ))) \
      vcat(RBVars.P.RHSₙ[1], zeros(T, RBVars.nₛᵖ, 1)))
  end

  RBVars.P.uₙ = xₙ[1:RBVars.P.nₛᵘ,:]
  RBVars.pₙ = xₙ[RBVars.P.nₛᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::StokesSteady)
  println("Reconstructing FEM solution from the newly computed RB one")
  reconstruct_FEM_solution(RBVars.P)
  RBVars.p̃ = RBVars.Φₛᵖ * RBVars.pₙ
end

function offline_phase(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady)

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
  RBVars::StokesSteady{T},
  param_nbs) where T

  μ = load_CSV(Array{T}[],
    joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(RBInfo.paths.mesh_path)
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)

  mean_H1_err = 0.0
  mean_L2_err = 0.0
  mean_pointwise_err_u = zeros(T, RBVars.P.Nₛᵘ)
  mean_pointwise_err_p = zeros(T, RBVars.Nₛᵖ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(RBInfo, RBVars)

  ũ_μ = zeros(T, RBVars.P.Nₛᵘ, length(param_nbs))
  uₙ_μ = zeros(T, RBVars.P.nₛᵘ, length(param_nbs))
  p̃_μ = zeros(T, RBVars.P.Nₛᵘ, length(param_nbs))
  pₙ_μ = zeros(T, RBVars.P.nₛᵘ, length(param_nbs))

  for nb in param_nbs
    println("Considering Parameter number: $nb")

    Param = get_ParamInfo(RBInfo, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"),
      DataFrame))[:, nb]
    pₕ_test = Matrix{T}(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "pₕ.csv"),
      DataFrame))[:, nb]

    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    mean_online_time = RBVars.P.online_time / length(param_nbs)
    mean_reconstruction_time = reconstruction_time / length(param_nbs)

    H1_err_nb = compute_errors(uₕ_test, RBVars, RBVars.P.Xᵘ₀)
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test - RBVars.P.ũ) / length(param_nbs)

    L2_err_nb = compute_errors(pₕ_test, RBVars, RBVars.Xᵖ₀)
    mean_L2_err += L2_err_nb / length(param_nbs)
    mean_pointwise_err_p += abs.(pₕ_test - RBVars.p̃) / length(param_nbs)

    ũ_μ[:, nb - param_nbs[1] + 1] = RBVars.P.ũ
    uₙ_μ[:, nb - param_nbs[1] + 1] = RBVars.P.uₙ
    p̃_μ[:, nb - param_nbs[1] + 1] = RBVars.p̃
    pₙ_μ[:, nb - param_nbs[1] + 1] = RBVars.pₙ

    println("Online wall time: $(RBVars.P.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-error: $H1_err_nb (snapshot number $nb)")
    println("Relative reconstruction L2-error: $L2_err_nb (snapshot number $nb)")

  end

  string_param_nbs = "params"
  for Param_nb in param_nbs
    string_param_nbs *= "_" * string(Param_nb)
  end
  path_μ = joinpath(RBInfo.paths.results_path, string_param_nbs)

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
      RBVars.P.S.offline_time = NaN
    end

    if !RBInfo.import_offline_structures
      times = Dict(RBVars.P.S.offline_time=>"off_time",
        mean_online_time=>"on_time", mean_reconstruction_time=>"rec_time")
    else
      times = Dict(mean_online_time=>"on_time",
        mean_reconstruction_time=>"rec_time")
    end
    CSV.write(joinpath(path_μ, "times.csv"),times)

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FEMSpace"=>FEMSpace,
    "mean_point_err_u"=>mean_pointwise_err_u,
    "mean_point_err_p"=>mean_pointwise_err_p)

  if RBInfo.post_process
    post_process(RBInfo, pass_to_pp)
  end

end
