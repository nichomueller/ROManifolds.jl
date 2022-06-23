function ROM_paths(root, problem_type, problem_name, mesh_name, RB_method, case)
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, case)
  mesh_path = paths.mesh_path
  FEM_snap_path = paths.FEM_snap_path
  FEM_structures_path = paths.FEM_structures_path
  ROM_path = joinpath(paths.current_test, RB_method)
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

function build_sparse_mat(
  FEMInfo::ProblemInfoSteady,
  FEMSpace::SteadyProblem,
  Param::ParametricInfoSteady,
  el::Vector{Float64};
  var="A")

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  if var == "A"
    Mat = assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α*∇(FEMSpace.ϕᵤ)))*dΩ_sparse,
      FEMSpace.V, FEMSpace.V₀)
  else
    error("Unrecognized sparse matrix")
  end

  Mat

end

function build_sparse_mat(
  FEMInfo::ProblemInfoUnsteady,
  FEMSpace::UnsteadyProblem,
  Param::ParametricInfoUnsteady,
  el::Vector{Float64},
  timesθ::Vector{Float64};
  var="A")

  Ω_sparse = view(FEMSpace.Ω, el)
  dΩ_sparse = Measure(Ω_sparse, 2 * FEMInfo.order)
  Nₜ = length(timesθ)

  function define_Matₜ(t::Real, var::String)
    if var == "A"
      return assemble_matrix(∫(∇(FEMSpace.ϕᵥ)⋅(Param.α(t)*∇(FEMSpace.ϕᵤ(t))))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    elseif mat == "M"
      return assemble_matrix(∫(FEMSpace.ϕᵥ*(Param.m(t)*FEMSpace.ϕᵤ(t)))*dΩ_sparse,
        FEMSpace.V(t), FEMSpace.V₀)
    else
      error("Unrecognized sparse matrix")
    end
  end
  Matₜ(t) = define_Matₜ(t, var)

  for (i_t,t) in enumerate(timesθ)
    i,j,v = findnz(Matₜ(t))
    if i_t == 1
      global Mat = sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ*Nₜ)
    else
      Mat[:,(i_t-1)*FEMSpace.Nₛᵘ+1:i_t*FEMSpace.Nₛᵘ] =
        sparse(i,j,v,FEMSpace.Nₛᵘ,FEMSpace.Nₛᵘ)
    end
  end

  Mat

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

function remove_small_entries(A::Array,tol=1e-15) ::Array
  A[A.<=tol].=0
  A
end

function compute_errors(uₕ::Matrix{Float64}, RBVars::RBSteadyProblem, norm_matrix = nothing)

  mynorm(uₕ - RBVars.ũ, norm_matrix) / mynorm(uₕ, norm_matrix)

end

function compute_errors(uₕ::Matrix{Float64}, RBVars::RBUnsteadyProblem, norm_matrix = nothing)

  H1_err = zeros(RBVars.Nₜ)
  H1_sol = zeros(RBVars.Nₜ)

  @simd for i = 1:RBVars.Nₜ
    H1_err[i] = mynorm(uₕ[:, i] - RBVars.S.ũ[:, i], norm_matrix)
    H1_sol[i] = mynorm(uₕ[:, i], norm_matrix)
  end

  return H1_err ./ H1_sol, norm(H1_err) / norm(H1_sol)

end

function compute_MDEIM_error(FEMSpace::FEMProblem, RBInfo::Info, RBVars::RBProblem)

  Aₙ_μ = (RBVars.Φₛᵘ)' * assemble_stiffness(FEMSpace, RBInfo, Param) * RBVars.Φₛᵘ

end

function post_process(root::String)

  println("Exporting plots and tables")

  function get_paths(dir::String)::Tuple
    path_to_err = joinpath(dir,
      "results/Params_95_96_97_98_99_100/H1L2_err.csv")
    path_to_t = joinpath(dir,
      "results/Params_95_96_97_98_99_100/times.csv")
    path_to_err,path_to_t
  end

  function get_tolerances(dir::String)::Vector{Float64}
    if occursin("-3",dir)
      return ["1e-3"]
    elseif occursin("-4",dir)
      return ["1e-4"]
    elseif occursin("-5",dir)
      return ["1e-5"]
    else
      return []
    end
  end

  function check_if_fun(dir::String,tol,tol_fun,err,err_fun,time,time_fun)
    path_to_err,path_to_t = get_paths(dir)
    if ispath(path_to_err) && ispath(path_to_t)
      ϵ = get_tolerances(dir)
      if !isempty(ϵ)
        if occursin("fun",dir)
          append!(tol_fun,ϵ)
          append!(err_fun,load_CSV(path_to_err)[1])
          cur_time = load_CSV(path_to_t)
          append!(time_fun["on"],cur_time[findall(x->x.=="on_time",cur_time[:,2]),1])
          append!(time_fun["off"],cur_time[findall(x->x.=="off_time",cur_time[:,2]),1])
        else
          append!(tol,ϵ)
          append!(err,load_CSV(path_to_err)[1])
          cur_time = load_CSV(path_to_t)
          append!(time["on"],cur_time[findall(x->x.=="on_time",cur_time[:,2]),1])
          append!(time["off"],cur_time[findall(x->x.=="off_time",cur_time[:,2]),1])
        end
      end
    end
    return tol,tol_fun,err,err_fun,time,time_fun
  end

  root_subs = get_all_subdirectories(root)
  filter!(el->!occursin("FEM_data",el),root_subs)

  (ϵ,ϵ_fun,ϵ_sampl,ϵ_fun_sampl,ϵ_nest,ϵ_fun_nest,ϵ_sampl_nest,ϵ_fun_sampl_nest) =
    (String[],String[],String[],String[],String[],String[],String[],String[])
  (errH1L2,errH1L2_fun,errH1L2_sampl,errH1L2_fun_sampl,errH1L2_nest,
    errH1L2_fun_nest,errH1L2_sampl_nest,errH1L2_fun_sampl_nest) =
    (Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],Float64[])
  (t,t_fun,t_sampl,t_fun_sampl,t_nest,t_fun_nest,t_sampl_nest,t_fun_sampl_nest) =
    (Dict("on"=>Float64[],"off"=>Float64[]),Dict("on"=>Float64[],"off"=>Float64[]),
    Dict("on"=>Float64[],"off"=>Float64[]),Dict("on"=>Float64[],"off"=>Float64[]),
    Dict("on"=>Float64[],"off"=>Float64[]),Dict("on"=>Float64[],"off"=>Float64[]),
    Dict("on"=>Float64[],"off"=>Float64[]),Dict("on"=>Float64[],"off"=>Float64[]))

  @simd for dir in root_subs
    if !occursin("nest",dir)
      if !occursin("sampl",dir)
        ϵ,ϵ_fun,errH1L2,errH1L2_fun,t,t_fun =
          check_if_fun(dir,ϵ,ϵ_fun,errH1L2,errH1L2_fun,t,t_fun)
      else
        ϵ_sampl,ϵ_fun_sampl,errH1L2_sampl,errH1L2_fun_sampl,t_sampl,t_fun_sampl =
          check_if_fun(dir,ϵ_sampl,ϵ_fun_sampl,errH1L2_sampl,errH1L2_fun_sampl,
          t_sampl,t_fun_sampl)
      end
    else
      if !occursin("sampl",dir)
        ϵ_nest,ϵ_fun_nest,errH1L2_nest,errH1L2_fun_nest,t_nest,t_fun_nest =
          check_if_fun(dir,ϵ_nest,ϵ_fun_nest,errH1L2_nest,errH1L2_fun_nest,
          t_nest,t_fun_nest)
      else
        ϵ_sampl_nest,ϵ_fun_sampl_nest,errH1L2_sampl_nest,errH1L2_fun_sampl_nest,
          t_sampl_nest,t_fun_sampl_nest = check_if_fun(dir,ϵ_sampl_nest,
          ϵ_fun_sampl_nest,errH1L2_sampl_nest,errH1L2_fun_sampl_nest,
          t_sampl_nest,t_fun_sampl_nest)
      end
    end
  end

  errors = Dict("Standard"=>errH1L2,"Functional"=>errH1L2_fun,
  "Standard-sampling"=>errH1L2_sampl,"Functional-sampling"=>errH1L2_fun_sampl,
  "Standard-nested"=>errH1L2_nest,"Functional-nested"=>errH1L2_fun_nest,
  "Standard-sampling-nested"=>errH1L2_sampl_nest,
  "Functional-sampling-nested"=>errH1L2_fun_sampl_nest)
  times = Dict("Standard"=>t,"Functional"=>t_fun,
  "Standard-sampling"=>t_sampl,"Functional-sampling"=>t_fun_sampl,
  "Standard-nested"=>t_nest,"Functional-nested"=>t_fun_nest,
  "Standard-sampling-nested"=>t_sampl_nest,
  "Functional-sampling-nested"=>t_fun_sampl_nest)
  tols = Dict("Standard"=>ϵ,"Functional"=>ϵ_fun,
  "Standard-sampling"=>ϵ_sampl,"Functional-sampling"=>ϵ_fun_sampl,
  "Standard-nested"=>ϵ_nest,"Functional-nested"=>ϵ_fun_nest,
  "Standard-sampling-nested"=>ϵ_sampl_nest,
  "Functional-sampling-nested"=>ϵ_fun_sampl_nest)

  plots_dir = joinpath(root,"plots")
  create_dir(plots_dir)

  for (key, val) in errors
    if !isempty(val)
      ϵ = parse.(Float64,tols[key])
      xvals = hcat(ϵ,ϵ)
      yvals = hcat(val,ϵ)
      generate_and_save_plot(xvals,yvals,"Average H¹-l² err, method: "*key,
      ["H¹-l² err","ϵ"],"ϵ","",plots_dir,true,true;var="err_"*key,
      selected_style=vcat(["lines"],["lines"]))
    end
  end
  CSV.write(joinpath(plots_dir, "errors.csv"),errors)
  CSV.write(joinpath(plots_dir, "times.csv"),times)
  #= for (key, val) in times
    generate_and_save_plot(val,"Average online time, method: "*key,tols[key],
    "","",plots_dir,false,true;var="time_"*key,selected_style=["scatter"])
  end =#

end
