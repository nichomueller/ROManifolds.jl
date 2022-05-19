include("../utils/general.jl")
include("../FEM/FEM_utils.jl")
include("RB_superclasses.jl")
include("M_DEIM.jl")

function ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim, RB_method, case; test_case="")
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, case)
  mesh_path = paths.mesh_path
  FEM_snap_path = paths.FEM_snap_path
  FEM_structures_path = paths.FEM_structures_path
  ROM_path = joinpath(paths.current_test, RB_method*test_case)
  create_dir(ROM_path)
  basis_path = joinpath(ROM_path, "basis")
  create_dir(basis_path)
  ROM_structures_path = joinpath(ROM_path, "ROM_structures")
  create_dir(ROM_structures_path)
  gen_coords_path = joinpath(ROM_path, "gen_coords")
  create_dir(gen_coords_path)
  results_path = joinpath(ROM_path, "results")
  create_dir(results_path)
  _ -> (mesh_path, FEM_snap_path, FEM_structures_path, basis_path, ROM_structures_path, gen_coords_path, results_path)
end

function POD(S, ϵ = 1e-5, X = nothing)

  S̃ = copy(S)

  if !isnothing(X)
    if !issparse(X)
      X = sparse(X)
    end

    H = cholesky(X)
    L = sparse(H.L)
    mul!(S̃, L', S̃[H.p, :])
  end

  if issparse(S̃)
    U, Σ, _ = svds(S̃; nsv=size(S̃)[2] - 1)[1]
  else
    U, Σ, _ = svd(S̃)
  end

  total_energy = sum(Σ .^ 2)
  cumulative_energy = 0.0
  N = 0

  while cumulative_energy / total_energy < 1.0 - ϵ ^ 2 && N < size(S̃)[2]
    N += 1
    cumulative_energy += Σ[N] ^ 2
    @info "POD loop number $N, cumulative energy = $cumulative_energy"
  end

  @info "Basis number obtained via POD is $N, projection error ≤ $((sqrt(abs(1 - cumulative_energy / total_energy))))"

  if issparse(U)
    U = Matrix(U)
  end

  if !isnothing(X)
    return Matrix((L' \ U[:, 1:N])[invperm(H.p), :]), Σ
  else
    return U[:, 1:N], Σ
  end

end

function build_sparse_mat(problem_info::ProblemSpecifics, ROM_info, μ_i::Array, el::Array; var="A")

  param = get_parametric_specifics(ROM_info, μ_i)
  FE_space = get_FESpace(problem_info, param.model)

  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * problem_info.order)
  if var === "A"
    Mat = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α * ∇(FE_space.ϕᵤ))) * dΩ_sparse, FE_space.V, FE_space.V₀)
  else
    @error "Unrecognized sparse matrix"
  end

  Mat

end

function build_sparse_mat(problem_info::ProblemSpecificsUnsteady, ROM_info, μ_i::Array, el::Array; var="A")

  param = get_parametric_specifics(ROM_info, μ_i)
  FE_space = get_FESpace(problem_info, param.model)

  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * problem_info.order)
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)

  function define_Matₜ(t::Real, var::String)
    if var === "A"
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)
    elseif mat === "M"
      return assemble_matrix(∫(FE_space.ϕᵥ ⋅ (param.m(t) * FE_space.ϕᵤ(t))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)
    else
      @error "Unrecognized sparse matrix"
    end
  end
  Matₜ(t) = define_Matₜ(t, var)

  i,j,v = findnz(Matₜ(times_θ[1]))
  Mat = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ*Nₜ)
  for (i_t,t) in enumerate(times_θ[2:end])
    i,j,v = findnz(Matₜ(t))
    Mat[:,i_t*FE_space.Nₛᵘ+1:(i_t+1)*FE_space.Nₛᵘ] = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ)
  end

  Mat

end

function build_sparse_mat(problem_info::ProblemSpecificsUnsteady, ROM_info, μ_i::Array, el::Array, time_idx::Array; var="A")

  param = get_parametric_specifics(ROM_info, μ_i)
  FE_space = get_FESpace(problem_info, param.model)

  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * problem_info.order)
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  times_MDEIM = times_θ[time_idx]
  Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)

  function define_Matₜ(t::Real, var::String)
    if var === "A"
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)
    elseif mat === "M"
      return assemble_matrix(∫(FE_space.ϕᵥ ⋅ (param.m(t) * FE_space.ϕᵤ(t))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)
    else
      @error "Unrecognized sparse matrix"
    end
  end
  Matₜ(t) = define_Matₜ(t, var)

  i,j,v = findnz(Matₜ(times_MDEIM[1]))
  Mat = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ*Nₜ)
  for (i_t,t) in enumerate(times_MDEIM[2:end])
    i,j,v = findnz(Matₜ(t))
    Mat[:,i_t*FE_space.Nₛᵘ+1:(i_t+1)*FE_space.Nₛᵘ] = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ)
  end

  Mat

end

function get_parametric_specifics(ROM_info::ROMSpecificsSteady, μ::Array)

  model = DiscreteModelFromFile(ROM_info.paths.mesh_path)

  function prepare_α(x, μ, probl_nl)
    if !probl_nl["A"]
      return sum(μ)
    else
      return 1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])
    end
  end
  α(x) = prepare_α(x, μ, ROM_info.probl_nl)

  function prepare_f(x, μ, probl_nl)
    if probl_nl["f"]
      return sin(μ[4] * x[1]) + sin(μ[4] * x[2])
    else
      return 1
    end
  end
  f(x) = prepare_f(x, μ, ROM_info.probl_nl)
  g(x) = 0
  h(x) = 1

  ParametricSpecifics(μ, model, α, f, g, h)

end

function get_parametric_specifics(ROM_info::ROMSpecificsUnsteady, μ::Array)

  model = DiscreteModelFromFile(ROM_info.paths.mesh_path)
  αₛ(x) = 1
  αₜ(t, μ) = sum(μ) * (2 + sin(2π * t))
  mₛ(x) = 1
  mₜ(t::Real) = 1
  m(x, t::Real) = mₛ(x)*mₜ(t)
  m(t::Real) = x -> m(x, t)
  fₛ(x) = 1
  fₜ(t::Real) = sin(π * t)
  gₛ(x) = 0
  gₜ(t::Real) = 0
  g(x, t::Real) = gₛ(x)*gₜ(t)
  g(t::Real) = x -> g(x, t)
  hₛ(x) = 0
  hₜ(t::Real) = 0
  h(x, t::Real) = hₛ(x)*hₜ(t)
  h(t::Real) = x -> h(x, t)
  u₀(x) = 0

  function prepare_α(x, t, μ, probl_nl)
    if !probl_nl["A"]
      return αₛ(x)*αₜ(t, μ)
    else
      return (1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) * sin(t) / μ[3]))
    end
  end
  α(x, t::Real) = prepare_α(x, t, μ, ROM_info.probl_nl)
  α(t::Real) = x -> α(x, t)

  function prepare_f(x, t, μ, probl_nl)
    if !probl_nl["f"]
      return fₛ(x)*fₜ(t)
    else
      return sin(π*t*x*(μ[4]+μ[5]))
    end
  end
  f(x, t::Real) = prepare_f(x, t, μ, ROM_info.probl_nl)
  f(t::Real) = x -> f(x, t)

  ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, g, hₛ, hₜ, h, u₀)

end

function assemble_online_structure(coeff, Mat::Array)

  Mat_μ = zeros(size(Mat)[1:end-1])
  dim = length(size(Mat))

  if dim === 2
    for q = eachindex(coeff)
      Mat_μ += Mat[:,q] * coeff[q]
    end
  elseif dim === 3
    for q = eachindex(coeff)
      Mat_μ += Mat[:,:,q] * coeff[q]
    end
  end

  Mat_μ

end

function blocks_to_matrix(A_block::Array, N_blocks::Int)

  A = zeros(prod(size(A_block[1])), N_blocks)
  for n = 1:N_blocks
    A[:, n] = A_block[n][:]
  end

  A

end

function matrix_to_blocks(A::Array)

  A_block = Matrix{Float64}[]
  N_blocks = size(A)[end]
  dims = Tuple(size(A)[1:end-1])
  order = prod(size(A)[1:end-1])
  for n = 1:N_blocks
    push!(A_block, reshape(A[:][(n-1)*order+1:n*order], dims))
  end

  A_block

end

function compute_errors(uₕ::Array, RB_variables::RBSteadyProblem, norm_matrix = nothing)

  mynorm(uₕ - RB_variables.ũ, norm_matrix) / mynorm(uₕ, norm_matrix)

end

function compute_errors(uₕ::Array, RB_variables::RBUnsteadyProblem, norm_matrix = nothing)

  H1_err = zeros(RB_variables.Nₜ)
  H1_sol = zeros(RB_variables.Nₜ)

  for i = 1:RB_variables.Nₜ
    H1_err[i] = mynorm(uₕ[:, i] - RB_variables.steady_info.ũ[:, i], norm_matrix)
    H1_sol[i] = mynorm(uₕ[:, i], norm_matrix)
  end

  return H1_err ./ H1_sol, norm(H1_err) / norm(H1_sol)

end

function compute_MDEIM_error(problem_info, ROM_info, RB_variables, μ)

  parametric_info = get_parametric_specifics(ROM_info, μ)
  FE_space = get_FESpace(problem_info, parametric_info.model)
  Aₙ_μ = (RB_variables.Φₛᵘ)' * assemble_stiffness(FE_space, ROM_info, parametric_info) * RB_variables.Φₛᵘ

end

function post_process(ROM_info::SteadyProblem, d::Dict)
  plotly()

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    p = plot(collect(1:length(MDEIM_Σ)), MDEIM_Σ, yaxis=:log, lw = 3, title = "Decay singular values, MDEIM")
    xlabel!("σ index")
    ylabel!("σ value")
    if ROM_info.save_results
      savefig(p, joinpath(ROM_info.paths.results_path, "plot_MDEIM_Σ.eps"))
    end
  end
  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_Σ.csv"))
    p = plot(collect(1:length(DEIM_Σ)), DEIM_Σ, yaxis=:log, lw = 3, title = "Decay singular values, DEIM")
    xlabel!("σ index")
    ylabel!("σ value")
    if ROM_info.save_results
      savefig(p, joinpath(ROM_info.paths.results_path, "plot_DEIM_Σ.eps"))
    end
  end

  FE_space = d["FE_space"]
  mean_point_err = load_CSV(joinpath(d["path_μ"], "mean_point_err.csv"))[:]
  if ROM_info.save_results
    writevtk(FE_space.Ω, joinpath(d["path_μ"], "mean_point_err"), cellfields = ["err"=> FEFunction(FE_space.V, mean_point_err)])
  else
    plot(collect(1:length(pointwise_err)), pointwise_err, yaxis=:log, lw = 3, title = "Average |uₕ - ũ|")
  end

end

function post_process(ROM_info::UnsteadyProblem, d::Dict)
  pyplot()

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    p = plot(collect(1:length(MDEIM_Σ)), MDEIM_Σ, yaxis=:log, lw = 3, title = "Decay singular values, MDEIM")
    xlabel!("σ index")
    ylabel!("σ value")
    savefig(p, joinpath(ROM_info.paths.results_path, "plot_MDEIM_Σ.eps"))
  end
  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_Σ.csv"))
    p = plot(collect(1:length(DEIM_Σ)), DEIM_Σ, yaxis=:log, lw = 3, title = "Decay singular values, DEIM")
    xlabel!("σ index")
    ylabel!("σ value")
    savefig(p, joinpath(ROM_info.paths.results_path, "plot_DEIM_Σ.eps"))
  end

  times = collect(ROM_info.t₀+ROM_info.δt:ROM_info.δt:ROM_info.T)
  FE_space = d["FE_space"]
  vtk_dir = joinpath(d["path_μ"], "vtk_folder")
  create_dir(vtk_dir)
  createpvd(joinpath(vtk_dir,"mean_point_err")) do pvd
    for (i,t) in enumerate(times)
      errₕt = FEFunction(FE_space.V(t), d["mean_point_err"][:,i])
      pvd[i] = createvtk(FE_space.Ω, joinpath(vtk_dir, "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
    end
  end

  p = plot(collect(1:length(d["mean_H1_err"])), d["mean_H1_err"], yaxis=:log, lw = 3, title = "Average ||uₕ(t) - ũ(t)||ₕ₁")
  xlabel!("time [s]")
  ylabel!("H¹ error")
  savefig(p, joinpath(d["path_μ"], "mean_H1_err.eps"))
  p = plot(collect(1:length(d["H1_L2_err"])), d["H1_L2_err"], yaxis=:log, lw = 3, title = "||uₕ - ũ||ₕ₁₋ₗ₂")
  xlabel!("param μ number")
  ylabel!("H¹-L² error")
  savefig(p, joinpath(d["path_μ"], "H1_L2_err.eps"))

end
