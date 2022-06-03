include("../utils/general.jl")
include("../FEM/FEM_utils.jl")
include("RB_superclasses.jl")
include("M_DEIM.jl")

function ROM_paths(root, problem_type, problem_name, mesh_name, RB_method, case; test_case="")
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, case)
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

function build_sparse_mat(problem_info::ProblemSpecifics, FE_space::SteadyProblem, param::ParametricSpecifics, el::Array; var="A")

  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * problem_info.order)
  if var == "A"
    Mat = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α * ∇(FE_space.ϕᵤ))) * dΩ_sparse, FE_space.V, FE_space.V₀)
  else
    @error "Unrecognized sparse matrix"
  end

  Mat

end

#= function build_sparse_mat(problem_info::ProblemSpecificsUnsteady,
  FE_space::UnsteadyProblem,
  ROM_info::Info,
  param::ParametricSpecificsUnsteady,
  el::Array;
  var="A")

  if !ROM_info.space_time_M_DEIM
    return build_sparse_mat_standard(problem_info::ProblemSpecificsUnsteady,
      FE_space::UnsteadyProblem,
      ROM_info::Info,
      param::ParametricSpecificsUnsteady,
      el::Array;
      var)
  else
    return build_sparse_mat_spacetime(problem_info::ProblemSpecificsUnsteady,
      FE_space::UnsteadyProblem,
      ROM_info::Info,
      param::ParametricSpecificsUnsteady,
      el::Array;
      var)
  end

end =#

function build_sparse_mat(problem_info::ProblemSpecificsUnsteady,
  FE_space::UnsteadyProblem,
  ROM_info::Info,
  param::ParametricSpecificsUnsteady,
  el::Vector;
  var="A")

  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * problem_info.order)
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  Nₜ = convert(Int64, ROM_info.T / ROM_info.δt)

  function define_Matₜ(t::Real, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)
    elseif mat == "M"
      return assemble_matrix(∫(FE_space.ϕᵥ ⋅ (param.m(t) * FE_space.ϕᵤ(t))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)
    else
      @error "Unrecognized sparse matrix"
    end
  end
  Matₜ(t) = define_Matₜ(t, var)

  for (i_t,t) in enumerate(times_θ)
    i,j,v = findnz(Matₜ(t))
    if i_t == 1
      global Mat = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ*Nₜ)
    else
      Mat[:,(i_t-1)*FE_space.Nₛᵘ+1:i_t*FE_space.Nₛᵘ] = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ)
    end
  end

  Mat

end

function build_sparse_mat(problem_info::ProblemSpecificsUnsteady,
  FE_space::UnsteadyProblem,
  ROM_info::Info,
  param::ParametricSpecificsUnsteady,
  el::Vector,
  MDEIM_idx::Vector;
  var="A")

  Ω_sparse = view(FE_space.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * problem_info.order)
  times_θ = collect(ROM_info.t₀:ROM_info.δt:ROM_info.T-ROM_info.δt).+ROM_info.δt*ROM_info.θ
  _, idx_time = from_spacetime_to_space_time_idx_vec(MDEIM_idx, FE_space.Nₛᵘ^2)
  idx_time = sort(idx_time)
  unique!(idx_time)

  function define_Matₜ(t::Real, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)
    elseif mat == "M"
      return assemble_matrix(∫(FE_space.ϕᵥ ⋅ (param.m(t) * FE_space.ϕᵤ(t))) * dΩ_sparse, FE_space.V(t), FE_space.V₀)
    else
      @error "Unrecognized sparse matrix"
    end
  end
  Matₜ(t) = define_Matₜ(t, var)

  for (i_t,t) in enumerate(times_θ[idx_time])
    i,j,v = findnz(Matₜ(t))
    if i_t == 1
      global Mat = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ*length(idx_time))
    else
      global Mat[:,(i_t-1)*FE_space.Nₛᵘ+1:i_t*FE_space.Nₛᵘ] = sparse(i,j,v,FE_space.Nₛᵘ,FE_space.Nₛᵘ)
    end
  end

  Mat

end

function assemble_online_structure(coeff, Mat::Matrix)

  Mat_μ = zeros(size(Mat)[1:end-1])
  dim = length(size(Mat))

  if dim == 2
    for q = eachindex(coeff)
      Mat_μ += Mat[:,q] * coeff[q]
    end
  elseif dim == 3
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
    H1_err[i] = mynorm(uₕ[:, i] - RB_variables.S.ũ[:, i], norm_matrix)
    H1_sol[i] = mynorm(uₕ[:, i], norm_matrix)
  end

  return H1_err ./ H1_sol, norm(H1_err) / norm(H1_sol)

end

function compute_MDEIM_error(FE_space::FEMProblem, ROM_info::Info, RB_variables::RBProblem)

  Aₙ_μ = (RB_variables.Φₛᵘ)' * assemble_stiffness(FE_space, ROM_info, parametric_info) * RB_variables.Φₛᵘ

end



function post_process(ROM_info::SteadyInfo, d::Dict)
  plotly()

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    make_plot(MDEIM_Σ, "Decay singular values, MDEIM", "σ index", "σ value", ROM_info.paths.results_path)
  end
  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_Σ.csv"))
    make_plot(DEIM_Σ, "Decay singular values, DEIM", "σ index", "σ value", ROM_info.paths.results_path)
  end

  FE_space = d["FE_space"]
  writevtk(FE_space.Ω, joinpath(d["path_μ"], "mean_point_err"), cellfields = ["err"=> FEFunction(FE_space.V, d["mean_point_err_u"])])

end

function post_process(ROM_info::UnsteadyInfo, d::Dict)

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    make_plot(MDEIM_Σ, "Decay singular values, MDEIM", "σ index", "σ value", ROM_info.paths.results_path)
  end
  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_Σ.csv"))
    make_plot(DEIM_Σ, "Decay singular values, DEIM", "σ index", "σ value", ROM_info.paths.results_path)
  end

  times = collect(ROM_info.t₀+ROM_info.δt:ROM_info.δt:ROM_info.T)
  FE_space = d["FE_space"]
  vtk_dir = joinpath(d["path_μ"], "vtk_folder")

  create_dir(vtk_dir)
  createpvd(joinpath(vtk_dir,"mean_point_err_u")) do pvd
    for (i,t) in enumerate(times)
      errₕt = FEFunction(FE_space.V(t), d["mean_point_err_u"][:,i])
      pvd[i] = createvtk(FE_space.Ω, joinpath(vtk_dir, "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
    end
  end

  make_plot(d["mean_H1_err"], "Average ||uₕ(t) - ũ(t)||ₕ₁", "time [s]", "H¹ error", d["path_μ"])
  make_plot(d["H1_L2_err"], "||uₕ - ũ||ₕ₁₋ₗ₂", "param μ number", "H¹-L² error", d["path_μ"])

  if length(keys(d)) == 8

    createpvd(joinpath(vtk_dir,"mean_point_err_p")) do pvd
      for (i,t) in enumerate(times)
        errₕt = FEFunction(FE_space.Q, d["mean_point_err_p"][:,i])
        pvd[i] = createvtk(FE_space.Ω, joinpath(vtk_dir, "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
      end
    end

    make_plot(d["mean_L2_err"], "Average ||pₕ(t) - p̃(t)||ₗ₂", "time [s]", "L² error", d["path_μ"])
    make_plot(d["L2_L2_err"], "||pₕ - p̃||ₗ₂₋ₗ₂", "param μ number", "L²-L² error", d["path_μ"])

  end

end

function make_plot(var::Vector, title::String, xlab::String, ylab::String, save_path::String)
  pyplot()
  p = plot(collect(1:length(var)), var, yaxis=:log, lw = 3, title = title)
  xlabel!(xlab)
  ylabel!(ylab)
  savefig(p, joinpath(save_path, string(var)*".eps"))
end
