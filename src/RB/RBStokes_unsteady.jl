include("RBPoisson_unsteady.jl")
include("ST-GRB_Stokes.jl")

function get_snapshot_matrix(RBInfo::Info, RBVars::StokesUnsteady)

  get_snapshot_matrix(RBInfo, RBVars.P)

  @info "Importing the snapshot matrix for field p, number of snapshots considered: $(RBInfo.nₛ)"
  Sᵖ = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "pₕ.csv"), DataFrame))[:, 1:(RBInfo.nₛ*RBVars.Nₜ)]
  RBVars.Sᵖ = Sᵖ
  RBVars.Nₛᵖ = size(Sᵖ)[1]
  RBVars.Nᵖ = RBVars.Nₛᵖ * RBVars.Nₜ
  @info "Dimension of snapshot matrix for field p: $(size(Sᵖ))"

end

function get_norm_matrix(RBInfo::Info, RBVars::PoissonSteady)

  if check_norm_matrix(RBVars)

    @info "Importing the norm matrices Xᵘ₀, Xᵘ, Xᵖ₀, Xᵖ"

    Xᵘ₀ = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "Xᵘ₀.csv"); convert_to_sparse = true)
    Xᵘ = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "Xᵘ.csv"); convert_to_sparse = true)
    Xᵖ₀ = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "Xᵖ₀.csv"); convert_to_sparse = true)
    Xᵖ = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "Xᵖ.csv"); convert_to_sparse = true)
    RBVars.Nₛᵘ = size(Xᵘ₀)[1]
    RBVars.Nᵖ = size(Xᵖ₀)[1]
    @info "Dimension of H¹ norm matrix, field u: $(size(Xᵘ₀))"
    @info "Dimension of L² norm matrix, field p: $(size(Xᵖ₀))"

    if RBInfo.use_norm_X
      RBVars.Xᵘ₀ = Xᵘ₀
      RBVars.Xᵘ = Xᵘ
      RBVars.Xᵖ₀ = Xᵖ₀
      RBVars.Xᵖ = Xᵖ
    else
      RBVars.Xᵘ₀ = I(RBVars.Nₛᵘ)
      RBVars.Xᵘ = I(RBVars.Nₛᵘ)
      RBVars.Xᵖ₀ = I(RBVars.Nₛᵖ)
      RBVars.Xᵖ = I(RBVars.Nₛᵖ)
    end

  end

end

function PODs_space(RBInfo::Info, RBVars::StokesUnsteady)

  get_norm_matrix(RBInfo, RBVars)
  PODs_space(RBInfo, RBVars.S)

  @info "Performing the nested spatial POD for fields (p,λ), using a tolerance of $(RBInfo.ϵₛ)"

  if RBInfo.perform_nested_POD

    for nₛ = 1:RBInfo.nₛ
      Sᵖₙ = RBVars.Sᵖ[:, (nₛ-1)*RBVars.P.Nₜ+1:nₛ*RBVars.P.Nₜ]
      Φₙᵖ, _ = POD(Sᵖₙ, RBInfo.ϵₛ)
      if nₛ ==1
        global Φₙᵖ_temp = Φₙᵖ
      else
        global Φₙᵖ_temp = hcat(Φₙᵖ_temp, Φₙᵖ)
      end
    end
    Φₛᵖ, _ = POD(Φₙᵖ_temp, RBInfo.ϵₛ)
    RBVars.Φₛᵖ = Φₛᵖ
    RBVars.nₛᵖ = size(Φₛᵖ)[2]

  else

    Φₛᵖ, _ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.Xᵖ₀)
    RBVars.Φₛᵖ = Φₛᵖ
    (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(Φₛᵖ)

  end

end

function PODs_time(RBInfo::Info, RBVars::StokesUnsteady)

  PODs_time(RBInfo, RBVars.S)

  @info "Performing the temporal POD for fields (p,λ), using a tolerance of $(RBInfo.ϵₜ)"

  if RBInfo.time_reduction_technique == "ST-HOSVD"
    Sᵖₜ = zeros(RBVars.P.Nₜ, RBVars.nₛᵖ * RBInfo.nₛ)
    Sᵖ = RBVars.Φₛᵖ' * RBVars.Sᵖ
    for i in 1:RBInfo.nₛ
      Sᵖₜ[:, (i-1)*RBVars.nₛᵖ+1:i*RBVars.nₛᵖ] =
      Sᵖ[:, (i-1)*RBVars.P.Nₜ+1:i*RBVars.P.Nₜ]'
    end
  else
    Sᵖₜ = zeros(RBVars.P.Nₜ, RBVars.Nₛᵖ * RBInfo.nₛ)
    Sᵖ = RBVars.Sᵖ
    for i in 1:RBInfo.nₛ
      Sᵖₜ[:, (i-1)*RBVars.Nₛᵖ+1:i*RBVars.Nₛᵖ] =
      transpose(Sᵖ[:, (i-1)*RBVars.P.Nₜ+1:i*RBVars.P.Nₜ])
    end
  end

  Φₜᵖ, _ = POD(Sᵖₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵖ = Φₜᵖ
  RBVars.nₜᵖ = size(Φₜᵖ)[2]

end

function import_reduced_basis(RBInfo::Info, RBVars::StokesUnsteady)

  import_reduced_basis(RBInfo, RBVars.P)

  @info "Importing the reduced basis for fields (p,λ)"

  RBVars.Φₛᵖ = load_CSV(joinpath(RBInfo.paths.basis_path, "Φₛᵖ.csv"))
  RBVars.nₛᵖ = size(RBVars.Φₛᵖ)[2]
  RBVars.Φₜᵖ = load_CSV(joinpath(RBInfo.paths.basis_path, "Φₜᵖ.csv"))
  RBVars.nₜᵖ = size(RBVars.Φₜᵖ)[2]
  RBVars.nᵖ = RBVars.nₛᵖ * RBVars.nₜᵖ

end

function index_mapping(i::Int, j::Int, RBVars::StokesUnsteady, var="u") :: Int64

  if var == "u"
    return index_mapping(i, j, RBVars.P)
  elseif var == "p"
    return convert(Int64, (i-1) * RBVars.nₜᵖ + j)
  else
    error("Unrecognized variable")
  end

end

function get_generalized_coordinates(RBInfo::Info, RBVars::StokesUnsteady, snaps=nothing)

  if check_norm_matrix(RBVars.P.S)
    get_norm_matrix(RBInfo, RBVars)
  end

  get_generalized_coordinates(RBInfo, RBVars.P)

  if isnothing(snaps) || maximum(snaps) > RBInfo.nₛ
    snaps = 1:RBInfo.nₛ
  end

  p̂ = zeros(RBVars.nᵖ, length(snaps))
  Φₛᵖ_normed = RBVars.Xᵖ₀ * RBVars.Φₛᵖ

  for (i, i_nₛ) = enumerate(snaps)

    @info "Assembling generalized coordinate relative to snapshot $(i_nₛ), field p"
    Sᵖ_i = RBVars.Sᵖ[:, (i_nₛ-1)*RBVars.P.Nₜ+1:i_nₛ*RBVars.P.Nₜ]
    for i_s = 1:RBVars.nₛᵖ
      for i_t = 1:RBVars.nₜᵖ
        Πᵖ_ij = reshape(Φₛᵖ_normed[:, i_s], :, 1) .* reshape(RBVars.Φₜᵖ[:, i_t], :, 1)'
        p̂[index_mapping(i_s, i_t, RBVars, "p"), i] = sum(Πᵖ_ij .* Sᵖ_i)
      end
    end

  end

  RBVars.p̂ = p̂

  if RBInfo.save_offline_structures
    save_CSV(p̂, joinpath(RBInfo.paths.gen_coords_path, "p̂.csv"))
  end

end

function test_offline_phase(RBInfo::Info, RBVars::StokesUnsteady)

  get_generalized_coordinates(RBInfo, RBVars, 1)

  uₙ = reshape(RBVars.P.S.û, (RBVars.P.nₜᵘ, RBVars.P.S.nₛᵘ))
  u_rec = RBVars.P.S.Φₛᵘ * (RBVars.P.Φₜᵘ * uₙ)'
  err = zeros(RBVars.P.Nₜ)
  for i = 1:RBVars.P.Nₜ
    err[i] = compute_errors(RBVars.P.S.Sᵘ[:, i], u_rec[:, i])
  end

end

function save_M_DEIM_structures(RBInfo::Info, RBVars::StokesUnsteady)

  save_M_DEIM_structures(RBInfo, RBVars.P)

end

function set_operators(RBInfo, RBVars::StokesUnsteady) :: Vector

  return vcat(["B"], set_operators(RBInfo, RBVars.P))

end


function get_M_DEIM_structures(RBInfo::Info, RBVars::StokesUnsteady) :: Vector

  get_M_DEIM_structures(RBInfo, RBVars.P)

end

function get_offline_structures(RBInfo::Info, RBVars::StokesUnsteady) :: Vector

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_Q(RBInfo::Info, RBVars::StokesUnsteady)

  get_Q(RBInfo, RBVars.P)

end

function solve_RB_system(RBInfo::Info, RBVars::StokesUnsteady, Param)

  get_RB_system(RBInfo, RBVars, Param)
  LHS_tmp = RBVars.P.S.LHSₙ
  RHS_tmp = RBVars.P.S.RHSₙ
  LHSₙ = [LHS_tmp[1] LHS_tmp[2]; LHS_tmp[3] LHS_tmp[4]]
  RHSₙ = [RHS_tmp[1]; RHS_tmp[2]]

  @info "Solving RB problem via backslash"
  @info "Condition number of the system's matrix: $(cond(LHSₙ))"
  xₙ = LHSₙ \ RHSₙ
  RBVars.P.S.uₙ = xₙ[1:RBVars.P.nᵘ,:]
  RBVars.pₙ = xₙ[RBVars.P.nᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::StokesUnsteady)

  reconstruct_FEM_solution(RBVars.P)

  pₙ = reshape(RBVars.pₙ, (RBVars.nₜᵖ, RBVars.nₛᵖ))
  RBVars.p̃ = RBVars.Φₛᵖ * (RBVars.Φₜᵖ * pₙ)'

end

function offline_phase(RBInfo::Info, RBVars::StokesUnsteady)

  RBVars.P.Nₜ = convert(Int64, RBInfo.T / RBInfo.δt)

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
    error("Impossible to assemble the reduced problem if neither
      the snapshots nor the bases can be loaded")
  end

  if import_snapshots_success && !import_basis_success
    @info "Failed to import the reduced basis, building it via POD"
    build_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.import_offline_structures

    operators = get_offline_structures(RBInfo, RBVars)
    if "A" ∈ operators || "B" ∈ operators || "M" ∈ operators || "MA" ∈ operators || "F" ∈ operators
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

end

function online_phase(RBInfo::Info, RBVars::StokesUnsteady, μ, Param_nbs)

  H1_L2_err = zeros(length(Param_nbs))
  L2_L2_err = zeros(length(Param_nbs))
  mean_H1_err = zeros(RBVars.Nₜ)
  mean_H1_L2_err = 0.0
  mean_L2_L2_err = 0.0
  mean_pointwise_err_u = zeros(RBVars.S.Nₛᵘ, RBVars.Nₜ)
  mean_pointwise_err_p = zeros(RBVars.S.Nₛᵖ, RBVars.Nₜ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(RBInfo, RBVars.S)

  ũ_μ = zeros(RBVars.S.Nₛᵘ, length(Param_nbs)*RBVars.Nₜ)
  uₙ_μ = zeros(RBVars.nᵘ, length(Param_nbs))
  p̃_μ = zeros(RBVars.S.Nₛᵘ, length(Param_nbs)*RBVars.Nₜ)
  pₙ_μ = zeros(RBVars.nᵘ, length(Param_nbs))

  for (i_nb, nb) in enumerate(Param_nbs)
    @info "Considering Parameter number: $nb"

    μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μ_nb)
    uₕ_test = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, (nb-1)*RBVars.P.Nₜ+1:nb*RBVars.P.Nₜ]
    pₕ_test = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "pₕ.csv"), DataFrame))[:, (nb-1)*RBVars.P.Nₜ+1:nb*RBVars.P.Nₜ]

    online_time = @elapsed begin
      solve_RB_system(RBInfo, RBVars, Param)
    end
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    mean_online_time = online_time / length(Param_nbs)
    mean_reconstruction_time = reconstruction_time / length(Param_nbs)

    H1_err_nb, H1_L2_err_nb, L2_err_nb, L2_L2_err_nb = compute_errors(uₕ_test, RBVars, RBVars.S.Xᵘ₀)

    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(Param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(Param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test - RBVars.S.ũ) / length(Param_nbs)
    ũ_μ[:, (i_nb-1)*RBVars.P.Nₜ+1:i_nb*RBVars.P.Nₜ] = RBVars.S.ũ
    uₙ_μ[:, i_nb] = RBVars.S.uₙ

    L2_L2_err[i_nb] = L2_L2_err_nb
    mean_L2_err += L2_err_nb / length(Param_nbs)
    mean_L2_L2_err += L2_L2_err_nb / length(Param_nbs)
    mean_pointwise_err_p += abs.(pₕ_test - RBVars.p̃) / length(Param_nbs)
    p̃_μ[:, (i_nb-1)*RBVars.P.Nₜ+1:i_nb*RBVars.P.Nₜ] = RBVars.S.p̃
    pₙ_μ[:, i_nb] = RBVars.S.pₙ

    @info "Online wall time: $online_time s (snapshot number $nb)"
    @info "Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)"
    @info "Relative reconstruction L2-L2 error: $L2_L2_err_nb (snapshot number $nb)"

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
    save_CSV(mean_pointwise_err_U, joinpath(path_μ, "mean_point_err_u.csv"))
    save_CSV(mean_H1_err, joinpath(path_μ, "H1_err.csv"))
    save_CSV([mean_H1_L2_err], joinpath(path_μ, "H1L2_err.csv"))

    save_CSV(p̃_μ, joinpath(path_μ, "p̃.csv"))
    save_CSV(Pₙ_μ, joinpath(path_μ, "Pₙ.csv"))
    save_CSV(mean_pointwise_err_p, joinpath(path_μ, "mean_point_err_p.csv"))
    save_CSV(mean_L2_err, joinpath(path_μ, "L2_err.csv"))
    save_CSV([mean_L2_L2_err], joinpath(path_μ, "L2L2_err.csv"))

    if !RBInfo.import_offline_structures
      times = Dict(RBVars.S.offline_time=>"off_time",
        mean_online_time=>"on_time", mean_reconstruction_time=>"rec_time")
    else
      times = Dict(mean_online_time=>"on_time",
        mean_reconstruction_time=>"rec_time")
    end
    CSV.write(joinpath(path_μ, "times.csv"),times)

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FEMSpace"=>FEMSpace,
    "H1_L2_err"=>H1_L2_err, "mean_H1_err"=>mean_H1_err,
    "mean_point_err_u"=>mean_pointwise_err_u,
    "L2_L2_err"=>L2_L2_err, "mean_L2_err"=>mean_L2_err,
    "mean_point_err_p"=>mean_pointwise_err_p)

  if RBInfo.postprocess
    post_process(RBInfo, pass_to_pp)
  end

end

function check_dataset(RBInfo, RBVars, i)

  μ = load_CSV(joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))
  μ_i = parse.(Float64, split(chop(μ[i]; head=1, tail=1), ','))
  Param = get_ParamInfo(problem_ntuple, RBInfo, μ_i)

  u1 = RBVars.S.Sᵘ[:, (i-1)*RBVars.P.Nₜ+1]
  u2 = RBVars.S.Sᵘ[:, (i-1)*RBVars.P.Nₜ+2]
  M = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "M.csv"); convert_to_sparse = true)
  A = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
  F = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "F.csv"))
  H = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "H.csv"))

  t¹_θ = RBInfo.t₀+RBInfo.δt*RBInfo.θ
  t²_θ = t¹_θ+RBInfo.δt

  LHS1 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t¹_θ,μ_i))
  RHS1 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t¹_θ)+H*Param.hₜ(t¹_θ))
  my_u1 = LHS1\RHS1

  LHS2 = RBInfo.θ*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t²_θ,μ_i))
  mat = (1-RBInfo.θ)*(M+RBInfo.δt*RBInfo.θ*A*Param.αₜ(t²_θ,μ_i))-M
  RHS2 = RBInfo.δt*RBInfo.θ*(F*Param.fₜ(t²_θ)+H*Param.hₜ(t²_θ))-mat*u1
  my_u2 = LHS2\RHS2

  u1≈my_u1
  u2≈my_u2

end

function compute_stability_constant(RBInfo, M, A, θ, Nₜ)

  #= M = assemble_mass(FEMSpace, RBInfo, Param)(0.0)
  A = assemble_stiffness(FEMSpace, RBInfo, Param)(0.0) =#
  Nₕ = size(M)[1]
  δt = RBInfo.T/Nₜ
  B₁ = θ*(M + θ*δt*A)
  B₂ = θ*(-M + (1-θ)*δt*A)
  λ₁,_ = eigs(B₁)
  λ₂,_ = eigs(B₂)

  return 1/(minimum(abs.(λ₁)) + minimum(abs.(λ₂)))

end
