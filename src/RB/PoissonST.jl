include("PoissonS.jl")
include("PoissonST_support.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T}) where T

  println("Importing the snapshot matrix for field u,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵘ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo),"uₕ.csv"),
    DataFrame))[:,1:RBInfo.nₛ*RBVars.Nₜ]

  RBVars.Sᵘ = Sᵘ
  RBVars.Nₛᵘ = size(Sᵘ)[1]
  RBVars.Nᵘ = RBVars.Nₛᵘ * RBVars.Nₜ

  println("Dimension of the snapshot matrix for field u: $(size(Sᵘ))")

end

function assemble_reduced_basis(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady)

  println("Building the space-time reduced basis for field u")

  RBVars.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
    PODs_time(RBInfo, RBVars)
  end

  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ
  RBVars.Nᵘ = RBVars.Nₛᵘ * RBVars.Nₜ

  if RBInfo.save_offline_structures
    save_CSV(RBVars.Φₛᵘ, joinpath(RBInfo.ROM_structures_path, "Φₛᵘ.csv"))
    save_CSV(RBVars.Φₜᵘ, joinpath(RBInfo.ROM_structures_path, "Φₜᵘ.csv"))
  end

  return

end

function get_reduced_basis(
  RBInfo::Info,
  RBVars::PoissonUnsteady{T}) where T

  get_reduced_basis(RBInfo, RBVars.Steady)

  println("Importing the temporal reduced basis for field u")
  RBVars.Φₜᵘ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₜᵘ.csv"))
  RBVars.nₜᵘ = size(RBVars.Φₜᵘ)[2]
  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ

end

function get_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady)

  operators = String[]

  append!(operators, get_A(RBInfo, RBVars))
  append!(operators, get_M(RBInfo, RBVars))

  if RBInfo.assemble_parametric_RHS
    append!(operators, get_F(RBInfo, RBVars))
    append!(operators, get_H(RBInfo, RBVars))
    append!(operators, get_L(RBInfo, RBVars))
  end

  operators

end

function assemble_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.offline_time += @elapsed begin
    for var ∈ intersect(operators, RBInfo.probl_nl)
      if var ∈ ("A", "M")
        assemble_MDEIM_matrices(RBInfo, RBVars, var)
      else
        assemble_DEIM_vectors(RBInfo, RBVars, var)
      end
    end

    for var ∈ setdiff(operators, RBInfo.probl_nl)
      if var ∈ ("A", "M")
        assemble_affine_matrices(RBInfo, RBVars, var)
      else
        assemble_affine_vectors(RBInfo, RBVars, var)
      end
    end
  end

  save_assembled_structures(RBInfo, RBVars)

end

function offline_phase(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady)

  println("Offline phase of the RB solver, unsteady Poisson problem")

  RBVars.Nₜ = Int(RBInfo.tₗ / RBInfo.δt)

  if RBInfo.get_snapshots
    get_snapshot_matrix(RBInfo, RBVars)
    get_snapshots_success = true
  else
    get_snapshots_success = false
  end

  if RBInfo.get_offline_structures
    get_reduced_basis(RBInfo, RBVars)
    get_basis_success = true
  else
    get_basis_success = false
  end

  if !get_snapshots_success && !get_basis_success
    error("Impossible to assemble the reduced problem if neither
      the snapshots nor the bases can be loaded")
  end

  if get_snapshots_success && !get_basis_success
    println("Failed to import the reduced basis, building it via POD")
    assemble_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.get_offline_structures
    operators = get_offline_structures(RBInfo, RBVars)
    if !isempty(operators)
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

end

################################## ONLINE ######################################

function get_θ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::UnsteadyParametricInfo) where T

  θᵃ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "A")
  θᵐ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "M")

  if !RBInfo.assemble_parametric_RHS
    θᶠ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "F")
    θʰ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "H")
    θˡ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "L")
  else
    θᶠ, θʰ, θˡ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  end

  return θᵃ, θᵐ, θᶠ, θʰ, θˡ

end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  θᵐ::Matrix{T},
  θᵃ::Matrix{T}) where T

  println("Assembling LHS using θ-method time scheme, θ=$(RBInfo.θ)")

  θ = RBInfo.θ
  δtθ = RBInfo.δt*θ
  nₜᵘ = RBVars.nₜᵘ
  Qᵐ = RBVars.Qᵐ
  Qᵃ = RBVars.Qᵃ

  Φₜᵘ_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ₁_M = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵐ)
  Φₜᵘ_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)
  Φₜᵘ₁_A = zeros(T,RBVars.nₜᵘ,RBVars.nₜᵘ,Qᵃ)

  @simd for i_t = 1:nₜᵘ
    for j_t = 1:nₜᵘ
      for q = 1:Qᵐ
        Φₜᵘ_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵐ[q,:])
        Φₜᵘ₁_M[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵐ[q,2:end])
      end
      for q = 1:Qᵃ
        Φₜᵘ_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵃ[q,:])
        Φₜᵘ₁_A[i_t,j_t,q] = sum(RBVars.Φₜᵘ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵃ[q,2:end])
      end
    end
  end

  Mₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Mₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵐ)
  Aₙ_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)
  Aₙ₁_tmp = zeros(T,RBVars.nᵘ,RBVars.nᵘ,Qᵃ)

  @simd for qᵐ = 1:Qᵐ
    Mₙ_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ_M[:,:,qᵐ])::Matrix{T}
    Mₙ₁_tmp[:,:,qᵐ] = kron(RBVars.Mₙ[:,:,qᵐ],Φₜᵘ₁_M[:,:,qᵐ])::Matrix{T}
  end
  @simd for qᵃ = 1:Qᵃ
    Aₙ_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ_A[:,:,qᵃ])::Matrix{T}
    Aₙ₁_tmp[:,:,qᵃ] = kron(RBVars.Aₙ[:,:,qᵃ],Φₜᵘ₁_A[:,:,qᵃ])::Matrix{T}
  end
  Mₙ = reshape(sum(Mₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Mₙ₁ = reshape(sum(Mₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ = δtθ*reshape(sum(Aₙ_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)
  Aₙ₁ = δtθ*reshape(sum(Aₙ₁_tmp,dims=3),RBVars.nᵘ,RBVars.nᵘ)

  block₁ = θ*(Aₙ+Mₙ) + (1-θ)*Aₙ₁ - θ*Mₙ₁
  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  θᶠ::Array{T},
  θʰ::Array{T},
  θˡ::Array{T}) where T

  println("Assembling RHS using θ-method time scheme, θ=$(RBInfo.θ)")

  Φₜᵘ_F = zeros(T, RBVars.nₜᵘ, RBVars.Qᶠ)
  Φₜᵘ_H = zeros(T, RBVars.nₜᵘ, RBVars.Qʰ)
  Φₜᵘ_L = zeros(T, RBVars.nₜᵘ, RBVars.Qˡ)

  @simd for i_t = 1:RBVars.nₜᵘ
    for q = 1:RBVars.Qᶠ
      Φₜᵘ_F[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θᶠ[q,:])
    end
    for q = 1:RBVars.Qʰ
      Φₜᵘ_H[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θʰ[q,:])
    end
    for q = 1:RBVars.Qˡ
      Φₜᵘ_L[i_t,q] = sum(RBVars.Φₜᵘ[:,i_t].*θˡ[q,:])
    end
  end

  block₁ = zeros(T, RBVars.nᵘ, 1)
  @simd for i_s = 1:RBVars.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ
      i_st = index_mapping(i_s, i_t, RBVars)
      Fₙ_μ_i_j = RBVars.Fₙ[i_s,:]'*Φₜᵘ_F[i_t,:]
      Hₙ_μ_i_j = RBVars.Hₙ[i_s,:]'*Φₜᵘ_H[i_t,:]
      Lₙ_μ_i_j = RBVars.Lₙ[i_s,:]'*Φₜᵘ_L[i_t,:]
      block₁[i_st, :] = Fₙ_μ_i_j + Hₙ_μ_i_j - Lₙ_μ_i_j
    end
  end

  block₁ *= RBInfo.δt*RBInfo.θ
  push!(RBVars.RHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady,
  Param::UnsteadyParametricInfo)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    blocks = [1]
    operators = get_system_blocks(RBInfo,RBVars,blocks,blocks)

    θᵃ, θᵐ, θᶠ, θʰ, θˡ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ)
    end

    if "RHS" ∈ operators
      if !RBInfo.assemble_parametric_RHS
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ, θˡ)
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,blocks,blocks,operators)

end

function solve_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady,
  Param::UnsteadyParametricInfo)

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")

  RBVars.online_time += @elapsed begin
    @fastmath RBVars.uₙ = RBVars.LHSₙ[1] \ RBVars.RHSₙ[1]
  end

end

function reconstruct_FEM_solution(RBVars::PoissonUnsteady)

  println("Reconstructing FEM solution from the newly computed RB one")
  uₙ = reshape(RBVars.uₙ, (RBVars.nₜᵘ, RBVars.nₛᵘ))
  @fastmath RBVars.ũ = RBVars.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'

end

function loop_on_params(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  μ::Vector{Vector{T}},
  param_nbs) where T

  H1_L2_err = zeros(T, length(param_nbs))
  mean_H1_err = zeros(T, RBVars.Nₜ)
  mean_H1_L2_err = 0.0
  mean_pointwise_err = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  ũ_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs)*RBVars.Nₜ)
  uₙ_μ = zeros(T, RBVars.nᵘ, length(param_nbs))
  mean_uₕ_test = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)

  for (i_nb, nb) in enumerate(param_nbs)
    println("\n")
    println("Considering parameter number: $nb/$(param_nbs[end])")

    Param = get_ParamInfo(RBInfo, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:,(nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]

    mean_uₕ_test += uₕ_test

    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    if i_nb > 1
      mean_online_time = RBVars.online_time/(length(param_nbs)-1)
      mean_reconstruction_time = reconstruction_time/(length(param_nbs)-1)
    end

    H1_err_nb, H1_L2_err_nb = compute_errors(
        RBVars, uₕ_test, RBVars.ũ, RBVars.Xᵘ₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err += abs.(uₕ_test-RBVars.ũ)/length(param_nbs)

    ũ_μ[:, (i_nb-1)*RBVars.Nₜ+1:i_nb*RBVars.Nₜ] = RBVars.ũ
    uₙ_μ[:, i_nb] = RBVars.uₙ

    println("Online wall time: $(RBVars.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)")
  end
  return (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,
    H1_L2_err,mean_online_time,mean_reconstruction_time)
end

function online_phase(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady,
  param_nbs) where T

  println("Online phase of the RB solver, unsteady Poisson problem")

  μ = load_CSV(Array{T}[],
    joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)

  get_norm_matrix(RBInfo, RBVars.Steady)
  (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,H1_L2_err,
    mean_online_time,mean_reconstruction_time) =
    loop_on_params(FEMSpace, RBInfo, RBVars, μ, param_nbs)

  adapt_time = 0.
  if RBInfo.adaptivity
    adapt_time = @elapsed begin
      (ũ_μ,uₙ_μ,_,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,
      H1_L2_err,_,_) =
      adaptive_loop_on_params(FEMSpace, RBInfo, RBVars, mean_uₕ_test,
      mean_pointwise_err, μ, param_nbs)
    end
  end

  string_param_nbs = "params"
  for Param_nb in param_nbs
    string_param_nbs *= "_" * string(Param_nb)
  end
  path_μ = joinpath(RBInfo.results_path, string_param_nbs)

  if RBInfo.save_results
    println("Saving the results...")
    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV(mean_H1_err, joinpath(path_μ, "H1_err.csv"))
    save_CSV(H1_L2_err, joinpath(path_μ, "H1L2_err.csv"))

    if RBInfo.get_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time+adapt_time,"rec_time"=>mean_reconstruction_time)
    CSV.write(joinpath(path_μ, "times.csv"),times)
  end

  pass_to_pp = Dict("path_μ"=>path_μ,
    "FEMSpace"=>FEMSpace, "H1_L2_err"=>H1_L2_err,
    "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>Float.(mean_pointwise_err))

  if RBInfo.post_process
    println("Post-processing the results...")
    post_process(RBInfo, pass_to_pp)
  end

end
