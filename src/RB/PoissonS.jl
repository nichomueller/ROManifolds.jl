include("RB.jl")
include("PoissonS_support.jl")

################################## ONLINE ######################################



function get_RB_LHS_blocks(
  RBVars::PoissonS{T},
  θᵃ::Vector{Vector{T}}) where T

  println("Assembling reduced LHS")

  block₁ = sum(Broadcasting(.*)(RBVars.Aₙ, θᵃ))
  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBVars::PoissonS{T},
  θᶠ::Vector{Vector{T}},
  θʰ::Vector{Vector{T}},
  θˡ::Vector{Vector{T}}) where T

  println("Assembling reduced RHS")

  mult = Broadcasting(.*)
  block₁ = sum(mult(RBVars.Fₙ, θᶠ)) + sum(mult(RBVars.Hₙ, θʰ)) - sum(mult(RBVars.Lₙ, θˡ))
  push!(RBVars.RHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_system(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::PoissonS,
  Param::ParamInfoS)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  blocks = [1]

  RBVars.online_time = @elapsed begin
    operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

    θᵃ, θᶠ, θʰ, θˡ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBVars, θᵃ)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_RHS
        get_RB_RHS_blocks(RBVars, θᶠ, θʰ, θˡ)
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,blocks,blocks,operators)

end

function solve_RB_system(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::PoissonS,
  Param::ParamInfoS)

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")
  RBVars.online_time += @elapsed begin
    RBVars.uₙ = RBVars.LHSₙ[1] \ RBVars.RHSₙ[1]
  end

end

function reconstruct_FEM_solution(RBVars::ROMInfoS)
  println("Reconstructing FEM solution from the newly computed RB one")
  RBVars.x̃ = Broadcasting(*)(RBVars.Φₛ, RBVars.xₙ)
end

function online_phase(
  RBInfo::ROMInfoS,
  RBVars::RBProblemS{T},
  param_nbs) where T

  function get_S_var(var::String, nb::Int)
    load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_snap_path(RBInfo), "$(var)ₕ.csv"))[:, nb]
  end

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)

  mean_err = T[]
  mean_pointwise_err = Vector{T}[]
  mean_online_time = 0.0

  get_norm_matrix(RBInfo, RBVars)

  for nb in param_nbs
    println("Considering parameter number: $nb")

    Param = get_ParamInfo(RBInfo, μ[nb])
    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruct_FEM_solution(RBVars)

    mean_online_time = RBVars.online_time / length(param_nbs)

    get_S(var) = get_S_var(var, nb)
    xₕ = Broadcasting(get_S)(RBInfo.vars)
    err_nb = Broadcasting(compute_errors)(xₕ, RBVars.x̃, RBVars.X₀)
    mean_err += err_nb / length(param_nbs)
    mean_pointwise_err += abs.(xₕ - RBVars.x̃) / length(param_nbs)

    println("Online wall time (snapshot number $nb): $(RBVars.online_time)s ")
  end

  string_param_nbs = "params"
  for param_nb in param_nbs
    string_param_nbs *= "_" * string(param_nb)
  end
  path_μ = joinpath(RBInfo.results_path, string_param_nbs)

  if RBInfo.save_results
    create_dir(path_μ)
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV(mean_err, joinpath(path_μ, "err.csv"))

    if RBInfo.get_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time, "on_time"=>mean_online_time)

    CSV.write(joinpath(path_μ, "times.csv"),times)
  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FEMSpace"=>FEMSpace,
    "mean_point_err_u"=>Float.(mean_pointwise_err))

  if RBInfo.post_process
    post_process(RBInfo, pass_to_pp)
  end

end
