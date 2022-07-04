function post_process(RBInfo::ROMInfoSteady, d::Dict) where T
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(MDEIM_Σ), MDEIM_Σ, "Decay singular values, MDEIM",
      ["σ"], "σ index", "σ value", RBInfo.paths.results_path; var="MDEIM_Σ")
  end
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(DEIM_Σ), DEIM_Σ, "Decay singular values, DEIM",
      ["σ"], "σ index", "σ value", RBInfo.paths.results_path; var="DEIM_Σ")
  end

  FEMSpace = d["FEMSpace"]
  writevtk(FEMSpace.Ω, joinpath(d["path_μ"], "mean_point_err"),
  cellfields = ["err"=> FEFunction(FEMSpace.V, d["mean_point_err_u"])])

end

function post_process(RBInfo::UnsteadyInfo, d::Dict)
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(MDEIM_Σ), MDEIM_Σ, "Decay singular values, MDEIM",
      ["σ"], "σ index", "σ value", RBInfo.paths.results_path; var="MDEIM_Σ")
  end
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(DEIM_Σ), DEIM_Σ, "Decay singular values, DEIM",
      ["σ"], "σ index", "σ value", RBInfo.paths.results_path; var="DEIM_Σ")
  end

  times = collect(RBInfo.t₀+RBInfo.δt:RBInfo.δt:RBInfo.tₗ)
  FEMSpace = d["FEMSpace"]
  vtk_dir = joinpath(d["path_μ"], "vtk_folder")

  create_dir(vtk_dir)
  createpvd(joinpath(vtk_dir,"mean_point_err_u")) do pvd
    for (i,t) in enumerate(times)
      errₕt = FEFunction(FEMSpace.V(t), d["mean_point_err_u"][:,i])
      pvd[i] = createvtk(FEMSpace.Ω, joinpath(vtk_dir,
        "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
    end
  end

  generate_and_save_plot(times,d["mean_H1_err"],
    "Average ||uₕ(t) - ũ(t)||ₕ₁", ["H¹ err"], "time [s]", "H¹ error", d["path_μ"];
    var="H1_err")
  xvec = collect(eachindex(d["H1_L2_err"]))
  generate_and_save_plot(xvec,d["H1_L2_err"],
    "||uₕ - ũ||ₕ₁₋ₗ₂", ["H¹-l² err"], "Param μ number", "H¹-l² error", d["path_μ"];
    var="H1_L2_err")

  if length(keys(d)) == 8

    createpvd(joinpath(vtk_dir,"mean_point_err_p")) do pvd
      for (i,t) in enumerate(times)
        errₕt = FEFunction(FEMSpace.Q, d["mean_point_err_p"][:,i])
        pvd[i] = createvtk(FEMSpace.Ω, joinpath(vtk_dir,
          "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
      end
    end

    generate_and_save_plot(times,d["mean_L2_err"],
      "Average ||pₕ(t) - p̃(t)||ₗ₂", ["l² err"], "time [s]", "L² error", d["path_μ"];
      var="L2_err")
    xvec = collect(eachindex(d["L2_L2_err"]))
    generate_and_save_plot(xvec,d["L2_L2_err"],
      "||pₕ - p̃||ₗ₂₋ₗ₂", ["l²-l² err"], "Param μ number", "L²-L² error", d["path_μ"];
      var="L2_L2_err")

  end

end

function plot_stability_constants(
  FEMSpace::FEMProblem,
  RBInfo::ROMInfoUnsteady,
  Param::ParametricInfoUnsteady)

  M = assemble_FEM_structure(FEMSpace, RBInfo, Param, "M")(0.0)
  A = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")(0.0)
  stability_constants = []
  for Nₜ = 10:10:1000
    const_Nₜ = compute_stability_constant(RBInfo,Nₜ,M,A)
    append!(stability_constants, const_Nₜ)
  end
  p = Plot.plot(collect(10:10:1000),
    stability_constants, xaxis=:log, yaxis=:log, lw = 3,
    label="||(Aˢᵗ)⁻¹||₂", title = "Euclidean norm of (Aˢᵗ)⁻¹", legend=:topleft)
  p = Plot.plot!(collect(10:10:1000), collect(10:10:1000),
    xaxis=:log, yaxis=:log, lw = 3, label="Nₜ")
  xlabel!("Nₜ")
  savefig(p, joinpath(RBInfo.paths.results_path, "stability_constant.eps"))

  function compute_stability_constant(RBInfo,Nₜ,M,A)
    δt = RBInfo.tₗ/Nₜ
    B₁ = RBInfo.θ*(M + RBInfo.θ*δt*A)
    B₂ = RBInfo.θ*(-M + (1-RBInfo.θ)*δt*A)
    λ₁,_ = eigs(B₁)
    λ₂,_ = eigs(B₂)
    return 1/(minimum(abs.(λ₁)) + minimum(abs.(λ₂)))
  end

end

function post_process(root::String)

  println("Exporting plots and tables")

  S = String
  T = Float64

  function get_paths(dir::String)
    path_to_err = joinpath(dir,
      "results/params_95_96_97_98_99_100/H1L2_err.csv")
    path_to_t = joinpath(dir,
      "results/params_95_96_97_98_99_100/times.csv")
    path_to_err,path_to_t
  end

  function get_tolerances(dir::String)
    if occursin("-3",dir)
      return ["1e-3"]
    elseif occursin("-4",dir)
      return ["1e-4"]
    elseif occursin("-5",dir)
      return ["1e-5"]
    else
      return [""]
    end
  end

  function check_if_fun(dir::String,tol,tol_fun,err,err_fun,time,time_fun)

    path_to_err,path_to_t = get_paths(dir)

    if ispath(path_to_err) && ispath(path_to_t)
      ϵ = get_tolerances(dir)
      if !isempty(ϵ)
        if occursin("fun",dir)
          append!(tol_fun,ϵ)
          append!(err_fun,load_CSV(Matrix{T}(undef,0,0), path_to_err)[1])
          cur_time = load_CSV(Matrix{T}(undef,0,0), path_to_t)
          append!(time_fun["on"],cur_time[findall(x->x.=="on_time",cur_time[:,2]),1])
          append!(time_fun["off"],cur_time[findall(x->x.=="off_time",cur_time[:,2]),1])
        else
          append!(tol,ϵ)
          append!(err,load_CSV(Matrix{T}(undef,0,0), path_to_err)[1])
          cur_time = load_CSV(Matrix{T}(undef,0,0), path_to_t)
          append!(time["on"],cur_time[findall(x->x.=="on_time",cur_time[:,2]),1])
          append!(time["off"],cur_time[findall(x->x.=="off_time",cur_time[:,2]),1])
        end
      end
    end

    tol,tol_fun,err,err_fun,time,time_fun

  end

  root_subs = get_all_subdirectories(root)
  filter!(el->!occursin("FEM_data",el),root_subs)

  (ϵ,ϵ_fun,ϵ_sampl,ϵ_fun_sampl,ϵ_nest,ϵ_fun_nest,ϵ_sampl_nest,ϵ_fun_sampl_nest) =
    (S[],S[],S[],S[],S[],S[],S[],S[])
  (errH1L2,errH1L2_fun,errH1L2_sampl,errH1L2_fun_sampl,errH1L2_nest,
    errH1L2_fun_nest,errH1L2_sampl_nest,errH1L2_fun_sampl_nest) =
    (T[],T[],T[],T[],T[],T[],T[],T[])
  (t,t_fun,t_sampl,t_fun_sampl,t_nest,t_fun_nest,t_sampl_nest,t_fun_sampl_nest) =
    (Dict("on"=>T[],"off"=>T[]),Dict("on"=>T[],"off"=>T[]),
    Dict("on"=>T[],"off"=>T[]),Dict("on"=>T[],"off"=>T[]),
    Dict("on"=>T[],"off"=>T[]),Dict("on"=>T[],"off"=>T[]),
    Dict("on"=>T[],"off"=>T[]),Dict("on"=>T[],"off"=>T[]))

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
      ϵ = parse.(T,tols[key])
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
