function post_process(::ROMInfoS, d::Dict) where T

  FEMSpace = d["FEMSpace"]
  writevtk(FEMSpace.Ω, joinpath(d["path_μ"], "mean_point_err_u"),
  cellfields = ["err"=> FEFunction(FEMSpace.V, d["mean_point_err_u"][:, 1])])

  if "mean_point_err_p" ∈ keys(d)
    FEMSpace = d["FEMSpace"]
    writevtk(FEMSpace.Ω, joinpath(d["path_μ"], "mean_point_err_p"),
    cellfields = ["err"=> FEFunction(FEMSpace.Q, d["mean_point_err_p"][:, 1])])
  end

end

function post_process(RBInfo::ROMInfoST{T}, d::Dict) where T

  times = collect(RBInfo.t₀+RBInfo.δt:RBInfo.δt:RBInfo.tₗ)
  FEMSpace = d["FEMSpace"]
  vtk_dir = joinpath(d["path_μ"], "vtk_folder")

  create_dir(vtk_dir)
  createpvd(joinpath(vtk_dir,"mean_point_err_u")) do pvd
    for (i,t) in enumerate(times)
      errₕt = FEFunction(FEMSpace.V(t), d["mean_point_err_u"][:,i])
      pvd[i] = createvtk(FEMSpace.Ω, joinpath(vtk_dir,
        "mean_point_err_u_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
    end
  end
  if "mean_point_err_p" ∈ keys(d)
    createpvd(joinpath(vtk_dir,"mean_point_err_p")) do pvd
      for (i,t) in enumerate(times)
        errₕt = FEFunction(FEMSpace.V(t), d["mean_point_err_p"][:,i])
        pvd[i] = createvtk(FEMSpace.Ω, joinpath(vtk_dir,
          "mean_point_err_p_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
      end
    end
  end

  generate_and_save_plot(times,d["mean_H1_err"],
    "Average ||uₕ(t) - ũ(t)||ₕ₁", ["H¹ err"], "time [s]", "H¹ error", d["path_μ"];
    var="H1_err")
  xvec = collect(eachindex(d["H1_L2_err"]))
  generate_and_save_plot(xvec,d["H1_L2_err"],
    "||uₕ - ũ||ₕ₁₋ₗ₂", ["H¹-l² err"], "Param μ number", "H¹-l² error", d["path_μ"];
    var="H1_L2_err", selected_style=["markers"])

  if "mean_L2_err" ∈ keys(d)

    generate_and_save_plot(times,d["mean_L2_err"],
      "Average ||pₕ(t) - p̃(t)||ₗ₂", ["l² err"], "time [s]", "L² error", d["path_μ"];
      var="L2_err")
    xvec = collect(eachindex(d["L2_L2_err"]))
    generate_and_save_plot(xvec,d["L2_L2_err"],
      "||pₕ - p̃||ₗ₂₋ₗ₂", ["l²-l² err"], "Param μ number", "L²-L² error", d["path_μ"];
      var="L2_L2_err", selected_style=["markers"])

  end

end

function plot_stability_constants(
  FEMSpace::FEMProblem,
  RBInfo::ROMInfoST,
  Param::ParamInfoST)

  function compute_stability_constant_Nₜ(RBInfo,Nₜ,M,A)
    println("Considering Nₜ = $Nₜ")
    δt = RBInfo.tₗ/Nₜ
    B₁ = RBInfo.θ*(M + RBInfo.θ*δt*A)
    λ₁,_ = eigs(B₁)
    return 1/minimum(abs.(λ₁))
  end

  function compute_stability_constant_μ(RBInfo,timesθ,M,A)
    λvec = zeros(length(timesθ))
    for (iₜ, t) = enumerate(timesθ)
      println("Considering time instant $iₜ/$(length(timesθ))")
      B₁ = RBInfo.θ*(M + RBInfo.θ*RBInfo.δt*A(t))
      λ,_ = eigs(Matrix(B₁))
      λvec[iₜ] = minimum(abs.(λ))
    end

    1/minimum(abs.(λvec))

  end

  M = assemble_FEM_structure(FEMSpace, RBInfo, Param, "M")(0.0)
  A(t) = assemble_FEM_structure(FEMSpace, RBInfo, Param, "A")(t)

  vec_Nₜ = collect(100:100:1000)
  stability_constants = []
  for Nₜ = vec_Nₜ
    const_Nₜ = compute_stability_constant_Nₜ(RBInfo,Nₜ,M,A(0.0))
    append!(stability_constants, const_Nₜ)
  end

  xval = hcat(vec_Nₜ,vec_Nₜ)
  yval = hcat(vec_Nₜ,stability_constants)
  label = ["Nₜ", "||(Aˢᵗ)⁻¹||₂"]
  paths = FEM_paths(root,problem_steadiness,problem_name,mesh_name,case)
  save_path = paths.current_test
  generate_and_save_plot(xval, yval, "Euclidean norm of (Aˢᵗ)⁻¹",
    label, "Nₜ", "||(Aˢᵗ)⁻¹||₂", save_path, true, true;
    var="stability_constant_Nₜ", selected_color=["black","blue"],
    selected_style=["lines","lines"], selected_dash = ["",""])

  timesθ = get_timesθ(RBInfo)
  compute_stability_constant_μ(RBInfo,timesθ,M,A)

end

function post_process(test_dir::String)

  println("Exporting plots and tables")

  S = String
  T = Float64

  function get_err_t_paths(res_path::String)
    param_path = get_all_subdirectories(res_path)[end]
    path_to_err = joinpath(param_path, "H1L2_err.csv")
    path_to_t = joinpath(param_path, "times.csv")
    path_to_err,path_to_t
  end

  function get_tolerances(dir::String)
    ["1e"*dir[end-1:end]]
  end

  function check_results(dir::String,tol,err,time)

    if ispath(joinpath(dir, "results"))

      path_to_err,path_to_t = get_err_t_paths(joinpath(dir, "results"))

      if ispath(path_to_err) && ispath(path_to_t)
        ϵ = get_tolerances(dir)
        if !isempty(ϵ)

          append!(tol,ϵ)
          append!(err,load_CSV(Matrix{T}(undef,0,0), path_to_err)[1])
          cur_time = load_CSV(Matrix(undef,0,0), path_to_t)
          append!(time["on"],cur_time[findall(x->x.=="on_time",cur_time[:,1]),2])
          append!(time["off"],cur_time[findall(x->x.=="off_time",cur_time[:,1]),2])

        end
      end

    end

    tol,err,time

  end

  function check_bases(dir::String, nₛᵘ, nₜᵘ, Qᵃ, Qᵃₜ, Qᶠ, Qᶠₜ, Qʰ, Qʰₜ)

    function check_basis_existence(path_to_bases::String, name::String)
      Q = NaN
      if isfile(joinpath(path_to_bases, name * ".csv"))
        M_DEIM_idx = load_CSV(Vector{Int}(undef,0),
          joinpath(path_to_bases, name * ".csv"))
        Q = length(M_DEIM_idx)
      end
      Q
    end

    path_to_bases = joinpath(dir, "ROM_structures")

    if ispath(path_to_bases)

      Φₛᵘ = load_CSV(Matrix{T}(undef,0,0),
        joinpath(path_to_bases, "Φₛᵘ.csv"))
      append!(nₛᵘ, size(Φₛᵘ)[2])

      Φₜᵘ = load_CSV(Matrix{T}(undef,0,0),
        joinpath(path_to_bases, "Φₜᵘ.csv"))
      append!(nₜᵘ, size(Φₜᵘ)[2])

      append!(Qᵃ, check_basis_existence(path_to_bases, "MDEIM_idx_A"))
      append!(Qᵃₜ, check_basis_existence(path_to_bases, "MDEIM_idx_time_A"))
      append!(Qᶠ, check_basis_existence(path_to_bases, "DEIM_idx_F"))
      append!(Qᶠₜ, check_basis_existence(path_to_bases, "DEIM_idx_time_F"))
      append!(Qʰ, check_basis_existence(path_to_bases, "DEIM_idx_H"))
      append!(Qʰₜ, check_basis_existence(path_to_bases, "DEIM_idx_time_H"))

    end

    nₛᵘ, nₜᵘ, Qᵃ, Qᵃₜ, Qᶠ, Qᶠₜ, Qʰ, Qʰₜ

  end

  function get_style_info(key::String, selected_color, selected_style, selected_dash)

    if occursin("functional", key)
      cur_color = "red"
    else
      cur_color = "blue"
    end

    if occursin("-ST", key)
      cur_dash = "dash"
    else
      cur_dash = ""
    end

    (vcat(selected_color, cur_color), vcat(selected_style, "lines"),
      vcat(selected_dash, cur_dash))

  end

  test_subdir = get_all_subdirectories(test_dir)
  filter!(el->!occursin("FEM_data",el),test_subdir)
  filter!(el->!occursin("plots",el),test_subdir)

  (ϵ,ϵ_fun,ϵ_st,ϵ_st_fun) = (S[],S[],S[],S[])
  (errH1L2,errH1L2_fun,errH1L2_st,errH1L2_st_fun) = (T[],T[],T[],T[])
  (t,t_fun,t_st,t_st_fun) =
    (Dict("on"=>T[],"off"=>T[]),Dict("on"=>T[],"off"=>T[]),
    Dict("on"=>T[],"off"=>T[]),Dict("on"=>T[],"off"=>T[]))

  nₛᵘ, nₜᵘ, Qᵃ, Qᵃₜ, Qᶠ, Qᶠₜ, Qʰ, Qʰₜ =
    Int[],Int[],Int[],Int[],Int[],Int[],Int[],Int[]
  nₛᵘ_fun, nₜᵘ_fun, Qᵃ_fun, Qᵃₜ_fun, Qᶠ_fun, Qᶠₜ_fun, Qʰ_fun, Qʰₜ_fun =
    Int[],Int[],Int[],Int[],Int[],Int[],Int[],Int[]
  nₛᵘ_st, nₜᵘ_st, Qᵃ_st, Qᵃₜ_st, Qᶠ_st, Qᶠₜ_st, Qʰ_st, Qʰₜ_st =
    Int[],Int[],Int[],Int[],Int[],Int[],Int[],Int[]
  (nₛᵘ_st_fun, nₜᵘ_st_fun, Qᵃ_st_fun, Qᵃₜ_st_fun, Qᶠ_st_fun, Qᶠₜ_st_fun,
    Qʰ_st_fun, Qʰₜ_st_fun) = Int[],Int[],Int[],Int[],Int[],Int[],Int[],Int[]

  for dir in test_subdir
    if !occursin("_st_",dir)
      if occursin("fun",dir)
        ϵ_fun,errH1L2_fun,t_fun = check_results(dir,ϵ_fun,errH1L2_fun,t_fun)
        nₛᵘ_fun, nₜᵘ_fun, Qᵃ_fun, Qᵃₜ_fun, Qᶠ_fun, Qᶠₜ_fun, Qʰ_fun, Qʰₜ_fun =
          check_bases(dir, nₛᵘ_fun, nₜᵘ_fun, Qᵃ_fun, Qᵃₜ_fun, Qᶠ_fun, Qᶠₜ_fun, Qʰ_fun, Qʰₜ_fun)
      else
        ϵ,errH1L2,t = check_results(dir,ϵ,errH1L2,t)
        nₛᵘ, nₜᵘ, Qᵃ, Qᵃₜ, Qᶠ, Qᶠₜ, Qʰ, Qʰₜ =
          check_bases(dir, nₛᵘ, nₜᵘ, Qᵃ, Qᵃₜ, Qᶠ, Qᶠₜ, Qʰ, Qʰₜ)
      end
    else
      if occursin("fun",dir)
        ϵ_st_fun,errH1L2_st_fun,t_st_fun =
          check_results(dir,ϵ_st_fun,errH1L2_st_fun,t_st_fun)
        nₛᵘ_st_fun, nₜᵘ_st_fun, Qᵃ_st_fun, Qᵃₜ_st_fun, Qᶠ_st_fun, Qᶠₜ_st_fun, Qʰ_st_fun, Qʰₜ_st_fun =
          check_bases(dir, nₛᵘ_st_fun, nₜᵘ_st_fun, Qᵃ_st_fun, Qᵃₜ_st_fun, Qᶠ_st_fun, Qᶠₜ_st_fun, Qʰ_st_fun, Qʰₜ_st_fun)
      else
        ϵ_st,errH1L2_st,t_st = check_results(dir,ϵ_st,errH1L2_st,t_st)
        nₛᵘ_st, nₜᵘ_st, Qᵃ_st, Qᵃₜ_st, Qᶠ_st, Qᶠₜ_st, Qʰ_st, Qʰₜ_st =
          check_bases(dir, nₛᵘ_st, nₜᵘ_st, Qᵃ_st, Qᵃₜ_st, Qᶠ_st, Qᶠₜ_st, Qʰ_st, Qʰₜ_st)
      end
    end
  end

  errors = Dict("err standard-S"=>errH1L2,"err functional-S"=>errH1L2_fun,
  "err standard-ST"=>errH1L2_st,"err functional-ST"=>errH1L2_st_fun)
  times = Dict("t standard-S"=>t,"t functional-S"=>t_fun,
  "t standard-ST"=>t_st,"t functional-ST"=>t_st_fun)
  tols = Dict("ϵ standard-S"=>ϵ,"ϵ functional-S"=>ϵ_fun,
  "ϵ standard-ST"=>ϵ_st,"ϵ functional-ST"=>ϵ_st_fun)

  n = Dict("nₛᵘ"=>nₛᵘ,"nₜᵘ"=>nₜᵘ,"Qᵃ"=>Qᵃ,"Qᶠ"=>Qᶠ,"Qʰ"=>Qʰ)
  n_st = ("nₛᵘ"=>nₛᵘ_st,"nₜᵘ"=>nₜᵘ_st,"Qᵃ"=>Qᵃ_st,"Qᵃₜ"=>Qᵃₜ_st,"Qᶠ"=>Qᶠ_st,"Qᶠₜ"=>Qᶠₜ_st,"Qʰ"=>Qʰ_st,"Qʰₜ"=>Qʰₜ_st)
  n_fun = ("nₛᵘ"=>nₛᵘ_fun,"nₜᵘ"=>nₜᵘ_fun,"Qᵃ"=>Qᵃ_fun,"Qᶠ"=>Qᶠ_fun,"Qʰ"=>Qʰ_fun)
  n_st_fun = ("nₛᵘ"=>nₛᵘ_st_fun,"nₜᵘ"=>nₜᵘ_st_fun,"Qᵃ"=>Qᵃ_st_fun,"Qᵃₜ"=>Qᵃₜ_st_fun,"Qᶠ"=>Qᶠ_st_fun,"Qᶠₜ"=>Qᶠₜ_st_fun,
    "Qʰ"=>Qʰ_st_fun,"Qʰₜ"=>Qʰₜ_st_fun)

  nbases = Dict("n"=>n,"n_st"=>n_st,"n_fun"=>n_fun,"n_st_fun"=>n_st_fun)

  plots_dir = joinpath(test_dir,"plots")
  create_dir(plots_dir)

  CSV.write(joinpath(plots_dir, "errors.csv"),errors)
  CSV.write(joinpath(plots_dir, "times.csv"),times)
  CSV.write(joinpath(plots_dir, "nbases.csv"),nbases)

  if ϵ == ϵ_fun == ϵ_st == ϵ_st_fun

    tol = parse.(T,ϵ)
    xlab, ylab = "ϵ", "H¹-l² err"

    xvals, yvals = tol, tol
    label, selected_style, selected_color, selected_dash = "ϵ", "lines", "black", ""
    for (key, val) in errors
      if !isempty(val)
        xvals = hcat(xvals, tol)
        yvals = hcat(yvals, val)
        label = vcat(label, key)
        selected_color, selected_style, selected_dash =
          get_style_info(key, selected_color, selected_style, selected_dash)
      end
    end

    generate_and_save_plot(xvals,yvals,"Average H¹-l² error",label,xlab,ylab,
      plots_dir,true,true;var="err",selected_color=selected_color,
      selected_style=selected_style,selected_dash=selected_dash)

  end

end
