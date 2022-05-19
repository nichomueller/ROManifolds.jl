include("RB_Poisson_unsteady.jl")
include("ST-GRB_Stokes.jl")
#include("ST-PGRB_Stokes.jl")

function get_snapshot_matrix(ROM_info::Info, RB_variables::StokesUnsteady)

  get_snapshot_matrix(ROM_info, RB_variables.P)

  @info "Importing the snapshot matrix for field p, number of snapshots considered: $(ROM_info.nₛ)"
  Sᵖ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, "pₕ.csv"), DataFrame))[:, 1:(ROM_info.nₛ*RB_variables.Nₜ)]
  RB_variables.Sᵖ = Sᵖ
  RB_variables.Nₛᵖ = size(Sᵖ)[1]
  RB_variables.Nᵖ = RB_variables.Nₛᵖ * RB_variables.Nₜ
  @info "Dimension of snapshot matrix for field p: $(size(Sᵖ))"

end

function get_norm_matrix(ROM_info::Info, RB_variables::PoissonSteady)

  if check_norm_matrix(RB_variables)

    @info "Importing the norm matrices Xᵘ₀, Xᵘ, Xᵖ₀, Xᵖ"

    Xᵘ₀ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Xᵘ₀.csv"); convert_to_sparse = true)
    Xᵘ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Xᵘ.csv"); convert_to_sparse = true)
    Xᵖ₀ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Xᵖ₀.csv"); convert_to_sparse = true)
    Xᵖ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Xᵖ.csv"); convert_to_sparse = true)
    RB_variables.Nₛᵘ = size(Xᵘ₀)[1]
    RB_variables.Nᵖ = size(Xᵖ₀)[1]
    @info "Dimension of H¹ norm matrix, field u: $(size(Xᵘ₀))"
    @info "Dimension of L² norm matrix, field p: $(size(Xᵖ₀))"

    if ROM_info.use_norm_X
      RB_variables.Xᵘ₀ = Xᵘ₀
      RB_variables.Xᵘ = Xᵘ
      RB_variables.Xᵖ₀ = Xᵖ₀
      RB_variables.Xᵖ = Xᵖ
    else
      RB_variables.Xᵘ₀ = I(RB_variables.Nₛᵘ)
      RB_variables.Xᵘ = I(RB_variables.Nₛᵘ)
      RB_variables.Xᵖ₀ = I(RB_variables.Nₛᵖ)
      RB_variables.Xᵖ = I(RB_variables.Nₛᵖ)
    end

  end

end

function PODs_space(ROM_info::Info, RB_variables::StokesUnsteady)

  get_norm_matrix(ROM_info, RB_variables)
  PODs_space(ROM_info, RB_variables.S)

  @info "Performing the nested spatial POD for fields (p,λ), using a tolerance of $(ROM_info.ϵₛ)"

  if ROM_info.perform_nested_POD

    for nₛ = 1:ROM_info.nₛ
      Sᵖₙ = RB_variables.Sᵖ[:, (nₛ-1)*RB_variables.P.Nₜ+1:nₛ*RB_variables.P.Nₜ]
      Φₙᵖ, _ = POD(Sᵖₙ, ROM_info.ϵₛ)
      if nₛ ===1
        global Φₙᵖ_temp = Φₙᵖ
      else
        global Φₙᵖ_temp = hcat(Φₙᵖ_temp, Φₙᵖ)
      end
    end
    Φₛᵖ, _ = POD(Φₙᵖ_temp, ROM_info.ϵₛ)
    RB_variables.Φₛᵖ = Φₛᵖ
    RB_variables.nₛᵖ = size(Φₛᵖ)[2]

  else

    Φₛᵖ, _ = POD(RB_variables.Sᵖ, ROM_info.ϵₛ, RB_variables.Xᵖ₀)
    RB_variables.Φₛᵖ = Φₛᵖ
    (RB_variables.Nₛᵖ, RB_variables.nₛᵖ) = size(Φₛᵖ)

  end

end

function PODs_time(ROM_info::Info, RB_variables::StokesUnsteady)

  PODs_time(ROM_info, RB_variables.S)

  @info "Performing the temporal POD for fields (p,λ), using a tolerance of $(ROM_info.ϵₜ)"

  if ROM_info.time_reduction_technique === "ST-HOSVD"
    Sᵖₜ = zeros(RB_variables.P.Nₜ, RB_variables.nₛᵖ * ROM_info.nₛ)
    Sᵖ = RB_variables.Φₛᵖ' * RB_variables.Sᵖ
    for i in 1:ROM_info.nₛ
      Sᵖₜ[:, (i-1)*RB_variables.nₛᵖ+1:i*RB_variables.nₛᵖ] =
      Sᵖ[:, (i-1)*RB_variables.P.Nₜ+1:i*RB_variables.P.Nₜ]'
    end
  else
    Sᵖₜ = zeros(RB_variables.P.Nₜ, RB_variables.Nₛᵖ * ROM_info.nₛ)
    Sᵖ = RB_variables.Sᵖ
    for i in 1:ROM_info.nₛ
      Sᵖₜ[:, (i-1)*RB_variables.Nₛᵖ+1:i*RB_variables.Nₛᵖ] =
      transpose(Sᵖ[:, (i-1)*RB_variables.P.Nₜ+1:i*RB_variables.P.Nₜ])
    end
  end

  Φₜᵖ, _ = POD(Sᵖₜ, ROM_info.ϵₜ)
  RB_variables.Φₜᵖ = Φₜᵖ
  RB_variables.nₜᵖ = size(Φₜᵖ)[2]

end

function import_reduced_basis(ROM_info::Info, RB_variables::StokesUnsteady)

  import_reduced_basis(ROM_info, RB_variables.P)

  @info "Importing the reduced basis for fields (p,λ)"

  RB_variables.Φₛᵖ = load_CSV(joinpath(ROM_info.paths.basis_path, "Φₛᵖ.csv"))
  RB_variables.nₛᵖ = size(RB_variables.Φₛᵖ)[2]
  RB_variables.Φₜᵖ = load_CSV(joinpath(ROM_info.paths.basis_path, "Φₜᵖ.csv"))
  RB_variables.nₜᵖ = size(RB_variables.Φₜᵖ)[2]
  RB_variables.nᵖ = RB_variables.nₛᵖ * RB_variables.nₜᵖ

end

function index_mapping(i::Int, j::Int, RB_variables::StokesUnsteady, var="u") :: Int64

  if var === "u"
    return index_mapping(i, j, RB_variables.P)
  elseif var === "p"
    return convert(Int64, (i-1) * RB_variables.nₜᵖ + j)
  else
    @error "Unrecognized variable"
  end

end

function get_generalized_coordinates(ROM_info::Info, RB_variables::StokesUnsteady, snaps=nothing)

  if check_norm_matrix(RB_variables.P.S)
    get_norm_matrix(ROM_info, RB_variables)
  end

  get_generalized_coordinates(ROM_info, RB_variables.P)

  if snaps === nothing || maximum(snaps) > ROM_info.nₛ
    snaps = 1:ROM_info.nₛ
  end

  p̂ = zeros(RB_variables.nᵖ, length(snaps))
  Φₛᵖ_normed = RB_variables.Xᵖ₀ * RB_variables.Φₛᵖ

  for (i, i_nₛ) = enumerate(snaps)

    @info "Assembling generalized coordinate relative to snapshot $(i_nₛ), field p"
    Sᵖ_i = RB_variables.Sᵖ[:, (i_nₛ-1)*RB_variables.P.Nₜ+1:i_nₛ*RB_variables.P.Nₜ]
    for i_s = 1:RB_variables.nₛᵖ
      for i_t = 1:RB_variables.nₜᵖ
        Πᵖ_ij = reshape(Φₛᵖ_normed[:, i_s], :, 1) .* reshape(RB_variables.Φₜᵖ[:, i_t], :, 1)'
        p̂[index_mapping(i_s, i_t, RB_variables, "p"), i] = sum(Πᵖ_ij .* Sᵖ_i)
      end
    end

  end

  RB_variables.p̂ = p̂

  if ROM_info.save_offline_structures
    save_CSV(p̂, joinpath(ROM_info.paths.gen_coords_path, "p̂.csv"))
  end

end

function test_offline_phase(ROM_info::Info, RB_variables::StokesUnsteady)

  get_generalized_coordinates(ROM_info, RB_variables, 1)

  uₙ = reshape(RB_variables.P.S.û, (RB_variables.P.nₜᵘ, RB_variables.P.S.nₛᵘ))
  u_rec = RB_variables.P.S.Φₛᵘ * (RB_variables.P.Φₜᵘ * uₙ)'
  err = zeros(RB_variables.P.Nₜ)
  for i = 1:RB_variables.P.Nₜ
    err[i] = compute_errors(RB_variables.P.S.Sᵘ[:, i], u_rec[:, i])
  end

end

function save_M_DEIM_structures(ROM_info::Info, RB_variables::StokesUnsteady)

  save_M_DEIM_structures(ROM_info, RB_variables.P)

end

function set_operators(ROM_info, RB_variables::StokesUnsteady) :: Vector

  return vcat(["B"], set_operators(ROM_info, RB_variables.P))

end


function get_M_DEIM_structures(ROM_info::Info, RB_variables::StokesUnsteady) :: Vector

  get_M_DEIM_structures(ROM_info, RB_variables.P)

end

function get_offline_structures(ROM_info::Info, RB_variables::StokesUnsteady) :: Vector

  operators = String[]
  append!(operators, get_affine_structures(ROM_info, RB_variables))
  append!(operators, get_M_DEIM_structures(ROM_info, RB_variables))
  unique!(operators)

  operators

end

function get_Q(ROM_info::Info, RB_variables::StokesUnsteady)

  get_Q(ROM_info, RB_variables.P)

end

function solve_RB_system(ROM_info::Info, RB_variables::StokesUnsteady, param)

  get_RB_system(ROM_info, RB_variables, param)
  LHS_tmp = RB_variables.P.S.LHSₙ
  RHS_tmp = RB_variables.P.S.RHSₙ
  LHSₙ = [LHS_tmp[1] LHS_tmp[2]; LHS_tmp[3] LHS_tmp[4]]
  RHSₙ = [RHS_tmp[1]; RHS_tmp[2]]

  @info "Solving RB problem via backslash"
  @info "Condition number of the system's matrix: $(cond(LHSₙ))"
  xₙ = LHSₙ \ RHSₙ
  RB_variables.P.S.uₙ = xₙ[1:RB_variables.P.nᵘ,:]
  RB_variables.pₙ = xₙ[RB_variables.P.nᵘ+1:end,:]

end

function reconstruct_FEM_solution(RB_variables::StokesUnsteady)

  reconstruct_FEM_solution(RB_variables.P)

  pₙ = reshape(RB_variables.pₙ, (RB_variables.nₜᵖ, RB_variables.nₛᵖ))
  RB_variables.p̃ = RB_variables.Φₛᵖ * (RB_variables.Φₜᵖ * pₙ)'

end

function build_RB_approximation(ROM_info::Info, RB_variables::StokesUnsteady)

  RB_variables.P.Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)

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
    if "A" ∈ operators || "B" ∈ operators || "M" ∈ operators || "MA" ∈ operators || "F" ∈ operators
      assemble_offline_structures(ROM_info, RB_variables, operators)
    end
  else
    assemble_offline_structures(ROM_info, RB_variables)
  end

end

function testing_phase(ROM_info::Info, RB_variables::StokesUnsteady, μ, param_nbs)

  H1_L2_err = zeros(length(param_nbs))
  L2_L2_err = zeros(length(param_nbs))
  mean_H1_err = zeros(RB_variables.Nₜ)
  mean_H1_L2_err = 0.0
  mean_L2_L2_err = 0.0
  mean_pointwise_err_u = zeros(RB_variables.S.Nₛᵘ, RB_variables.Nₜ)
  mean_pointwise_err_p = zeros(RB_variables.S.Nₛᵖ, RB_variables.Nₜ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(ROM_info, RB_variables.S)

  ũ_μ = zeros(RB_variables.S.Nₛᵘ, length(param_nbs)*RB_variables.Nₜ)
  uₙ_μ = zeros(RB_variables.nᵘ, length(param_nbs))
  p̃_μ = zeros(RB_variables.S.Nₛᵘ, length(param_nbs)*RB_variables.Nₜ)
  pₙ_μ = zeros(RB_variables.nᵘ, length(param_nbs))

  for (i_nb, nb) in enumerate(param_nbs)
    @info "Considering parameter number: $nb"

    μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_nb)
    uₕ_test = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, (nb-1)*RB_variables.P.Nₜ+1:nb*RB_variables.P.Nₜ]
    pₕ_test = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, "pₕ.csv"), DataFrame))[:, (nb-1)*RB_variables.P.Nₜ+1:nb*RB_variables.P.Nₜ]

    online_time = @elapsed begin
      solve_RB_system(ROM_info, RB_variables, parametric_info)
    end
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RB_variables)
    end
    mean_online_time = online_time / length(param_nbs)
    mean_reconstruction_time = reconstruction_time / length(param_nbs)

    H1_err_nb, H1_L2_err_nb, L2_err_nb, L2_L2_err_nb = compute_errors(uₕ_test, RB_variables, RB_variables.S.Xᵘ₀)

    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test - RB_variables.S.ũ) / length(param_nbs)
    ũ_μ[:, (i_nb-1)*RB_variables.P.Nₜ+1:i_nb*RB_variables.P.Nₜ] = RB_variables.S.ũ
    uₙ_μ[:, i_nb] = RB_variables.S.uₙ

    L2_L2_err[i_nb] = L2_L2_err_nb
    mean_L2_err += L2_err_nb / length(param_nbs)
    mean_L2_L2_err += L2_L2_err_nb / length(param_nbs)
    mean_pointwise_err_p += abs.(pₕ_test - RB_variables.p̃) / length(param_nbs)
    p̃_μ[:, (i_nb-1)*RB_variables.P.Nₜ+1:i_nb*RB_variables.P.Nₜ] = RB_variables.S.p̃
    pₙ_μ[:, i_nb] = RB_variables.S.pₙ

    @info "Online wall time: $online_time s (snapshot number $nb)"
    @info "Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)"
    @info "Relative reconstruction L2-L2 error: $L2_L2_err_nb (snapshot number $nb)"

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
    save_CSV(mean_pointwise_err_U, joinpath(path_μ, "mean_point_err_u.csv"))
    save_CSV(mean_H1_err, joinpath(path_μ, "H1_err.csv"))
    save_CSV([mean_H1_L2_err], joinpath(path_μ, "H1L2_err.csv"))

    save_CSV(p̃_μ, joinpath(path_μ, "p̃.csv"))
    save_CSV(Pₙ_μ, joinpath(path_μ, "Pₙ.csv"))
    save_CSV(mean_pointwise_err_p, joinpath(path_μ, "mean_point_err_p.csv"))
    save_CSV(mean_L2_err, joinpath(path_μ, "L2_err.csv"))
    save_CSV([mean_L2_L2_err], joinpath(path_μ, "L2L2_err.csv"))

    if !ROM_info.import_offline_structures
      times = [RB_variables.S.offline_time, mean_online_time, mean_reconstruction_time]
    else
      times = [mean_online_time, mean_reconstruction_time]
    end
    save_CSV(times, joinpath(path_μ, "times.csv"))

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FE_space"=>FE_space, "H1_L2_err"=>H1_L2_err, "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>mean_pointwise_err_u, "L2_L2_err"=>L2_L2_err, "mean_L2_err"=>mean_L2_err, "mean_point_err_p"=>mean_pointwise_err_p)

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

  u1 = RB_variables.S.Sᵘ[:, (i-1)*RB_variables.P.Nₜ+1]
  u2 = RB_variables.S.Sᵘ[:, (i-1)*RB_variables.P.Nₜ+2]
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
