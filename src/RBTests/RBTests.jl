include("FunctionDefinitions.jl")

function fem_path(tpath::String)
  create_dir!(tpath)
  fepath = joinpath(tpath,"fem")
  create_dir!(fepath)
  fepath
end

function rom_path(tpath::String,ϵ::Float)
  rbpath = joinpath(tpath,"$ϵ")
  create_dir!(rbpath)
  rbpath
end

function rom_offline_path(tpath::String,ϵ::Float)
  rb_off_path = joinpath(rom_path(tpath,ϵ),"offline")
  create_dir!(rb_off_path)
  rb_off_path
end

function rom_online_path(tpath::String,ϵ::Float)
  rb_on_path = joinpath(rom_path(tpath,ϵ),"online")
  create_dir!(rb_on_path)
  rb_on_path
end

function rom_off_on_paths(
  tpath::String,ϵ::Float;
  st_mdeim=false,fun_mdeim=false)

  @assert isdir(tpath) "Provide valid path for the current test"
  function keyword()
    if !st_mdeim && !fun_mdeim
      return "standard"
    else
      st = st_mdeim ? "st" : ""
      fun = fun_mdeim ? "fun" : ""
      return st*fun
    end
  end

  rompath = joinpath(tpath,"rom")
  keytpath = joinpath(rompath,keyword())

  offpath = rom_offline_path(keytpath,ϵ)
  onpath = rom_online_path(keytpath,ϵ)
  offpath,onpath
end

function mesh_path(tpath::String,mesh::String)
  joinpath(get_parent_dir(tpath;nparent=3),"meshes/$mesh")
end

function set_labels!(model,bnd_info)
  tags = collect(keys(bnd_info))
  bnds = collect(values(bnd_info))
  @assert length(tags) == length(bnds)
  labels = get_face_labeling(model)
  for i = eachindex(tags)
    if tags[i] ∉ labels.tag_to_name
      add_tag_from_tags!(labels,tags[i],bnds[i])
    end
  end
end

function model_info(
  mshpath::String,
  bnd_info::Dict,
  ::Val{false})

  model = DiscreteModelFromFile(mshpath)
  set_labels!(model,bnd_info)
  model
end

function model_info(
  ::String,
  bnd_info::Dict,
  ::Val{true})

  function model(μ) end
  set_labels!(model,bnd_info)
  model
end

function model_info(
  mshpath::String,
  bnd_info::Dict,
  ptype::ProblemType)

  model_info(mshpath,bnd_info,ispdomain(ptype))
end

function fe_snapshots(
  solver,
  op,
  fepath::String,
  run_fem::Bool,
  nsnap::Int,
  args...;
  indef=true,kwargs...)

  isindef = Val{indef}()
  if run_fem
    printstyled("Generating $nsnap full order snapshots\n";color=:blue)
    generate_fe_snapshots(isindef,solver,op,fepath,nsnap,args...;kwargs...)
  else
    printstyled("Loading $nsnap full order snapshots\n";color=:blue)
    load_fe_snapshots(isindef,fepath,nsnap)
  end
end

function load_fe_snapshots(::Val{false},fepath::String,nsnap::Int)
  load(fepath,:u,nsnap),load(Vector{Param},fepath)
end

function load_fe_snapshots(::Val{true},fepath::String,nsnap::Int)
  load(fepath,:u,nsnap),load(fepath,:p,nsnap),load(Vector{Param},fepath)
end

function generate_fe_snapshots(
  isindef::Val,
  solver::FESolver,
  op::ParamFEOperator,
  fepath::String,
  nsnap::Int;
  kwargs...)

  sol = solve(solver,op,nsnap)
  generate_fe_snapshots(isindef,sol,fepath;kwargs...)
end

function generate_fe_snapshots(
  isindef::Val,
  solver::ThetaMethod,
  op::ParamTransientFEOperator,
  fepath::String,
  nsnap::Int,
  t0::Real,
  tF::Real;
  kwargs...)

  sol = solve(solver,op,t0,tF,nsnap)
  generate_fe_snapshots(isindef,sol,fepath;kwargs...)
end

function generate_fe_snapshots(
  ::Val{false},
  sol::Vector{T},
  fepath::String;
  save_snap=true) where T

  ns = length(sol)
  time = @elapsed begin
    uh,μ = collect_solutions(sol)
  end
  usnap = Snapshots(:u,uh,ns)
  if save_snap
    save.((fepath,fepath),(usnap,μ))
    save(joinpath(fepath,"fe_time"),Dict("FE time"=>time))
  end
  usnap,μ
end

function generate_fe_snapshots(
  ::Val{true},
  sol::Vector{T},
  fepath::String;
  save_snap=true) where T

  Ns,ns = get_Ns(sol),length(sol)
  time = @elapsed begin
    xh,μ = collect_solutions(sol)
  end
  uh,ph = xh[1:Ns[1],:],xh[Ns[1]+1:Ns[1]+Ns[2],:]
  usnap,psnap = Snapshots(:u,uh,ns),Snapshots(:p,ph,ns)
  if save_snap
    save.((fepath,fepath,fepath),(usnap,psnap,μ))
    save(joinpath(fepath,"fe_time"),Dict("FE time"=>time))
  end
  usnap,psnap,μ
end

function collect_solutions(sol)
  Ns = sum(get_Ns(first(sol)))
  Nt = get_Nt(first(sol))
  ns = length(sol)

  x = Elemental.zeros(EMatrix{Float},Ns,Nt*ns)
  xtmp = Elemental.zeros(EMatrix{Float},Ns,Nt)
  μ = Param[]
  @sync @distributed for k in eachindex(sol)
    printstyled("Collecting solution $k\n";color=:blue)
    copyto!(view(x,:,(k-1)*Nt+1:k*Nt),get_solution(xtmp,sol[k]))
    push!(μ,sol[k].psol.μ)
  end

  x,μ
end

function get_solution(x::EMatrix{Float},solk::ParamFESolution)
  x[:,1] = get_free_dof_values(solk.psol.uh)
  x
end

function get_solution(x::EMatrix{Float},solk::ParamTransientFESolution)
  n = 1
  for (xn,tn) in solk
    printstyled("Time $tn\n";color=:blue)
    x[:,n] = xn
    n += 1
  end
  x
end

function get_dirichlet_values(
  U::ParamTrialFESpace,
  μ::Vector{Param})

  dir(μ) = U(μ).dirichlet_values
  Snapshots(:g,dir.(μ))
end

function get_dirichlet_values(
  U::ParamTransientTrialFESpace,
  μ::Vector{Param},
  tinfo::TimeInfo)

  times = get_times(tinfo)
  dir(μ) = Matrix([U(μ,t).dirichlet_values for t=times])
  Snapshots(:g,dir.(μ))
end
