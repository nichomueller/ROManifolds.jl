include("FunctionDefinitions.jl")

test_path(root,mesh,::Val{true}) = joinpath(root,"steady/$mesh")
test_path(root,mesh,::Val{false}) = joinpath(root,"unsteady/$mesh")

function fem_path(ptype::ProblemType,mesh::String,root::String)
  @assert isdir(root) "Provide valid root path"
  tpath = test_path(root,mesh,issteady(ptype))
  fepath = joinpath(tpath,"fem")
  create_dir!(fepath)
  fepath
end

function rom_path(tpath::String,ϵ::Float)
  rbpath = joinpath(joinpath(tpath,"rom"),"$ϵ")
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

function rom_off_on_paths(ptype::ProblemType,mesh::String,root::String,ϵ::Float;
  st_mdeim=false,fun_mdeim=false)

  @assert isdir(root) "Provide valid root path"
  function keyword()
    if !st_mdeim && !fun_mdeim
      return "standard"
    else
      st = st_mdeim ? "st" : ""
      fun = fun_mdeim ? "fun" : ""
      return st*fun
    end
  end

  tpath = test_path(root,mesh,issteady(ptype))
  new_tpath = joinpath(tpath,keyword())

  offpath = rom_offline_path(new_tpath,ϵ)
  onpath = rom_online_path(new_tpath,ϵ)
  offpath,onpath
end

function mesh_path(mesh::String,root::String)
  joinpath(get_parent_dir(root),"meshes/$mesh")
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
  ptype::ProblemType,
  solver,
  op,
  fepath::String,
  run_fem::Bool,
  nsnap::Int,
  args...;kwargs...)

  if run_fem
    println("Generating $nsnap full order snapshots")
    generate_fe_snapshots(ptype,solver,op,fepath,nsnap,args...)
  else
    println("Loading $nsnap full order snapshots")
    load_fe_snapshots(ptype,fepath,nsnap;kwargs...)
  end
end

function load_fe_snapshots(
  ptype::ProblemType,
  fepath::String,
  nsnap::Int;
  get_lift=false)

  snaps = load_fe_snapshots(isindef(ptype),fepath,nsnap)
  if get_lift
    snaps...,load_snap(fepath,:g,nsnap)
  else
    snaps
  end
end

function load_fe_snapshots(::Val{false},fepath::String,nsnap::Int)
  load_snap(fepath,:u,nsnap),load_param(fepath)
end

function load_fe_snapshots(::Val{true},fepath::String,nsnap::Int)
  load_snap(fepath,:u,nsnap),load_snap(fepath,:p,nsnap),load_param(fepath)
end

function generate_fe_snapshots(
  ptype::ProblemType,
  solver::FESolver,
  op::ParamFEOperator,
  fepath::String,
  nsnap::Int)

  sol = solve(solver,op,nsnap)
  generate_fe_snapshots(isindef(ptype),sol,fepath)
end

function generate_fe_snapshots(
  ptype::ProblemType,
  solver::ThetaMethod,
  op::ParamTransientFEOperator,
  fepath::String,
  nsnap::Int,
  t0::Real,
  tF::Real)

  sol = solve(solver,op,t0,tF,nsnap)
  generate_fe_snapshots(isindef(ptype),sol,fepath)
end

function generate_fe_snapshots(::Val{false},sol,fepath::String)
  time = @elapsed begin
    uh,gh,μ = collect_solutions(sol)
  end
  usnap,gsnap = Snapshots(:u,uh),Snapshots(:g,gh)
  save.((fepath,fepath,fepath),(usnap,gsnap,μ))
  save(fepath,Dict("FE time"=>time))
  usnap,μ,gsnap
end

function generate_fe_snapshots(::Val{true},sol,fepath::String)
  Ns = get_Ns(sol)
  time = @elapsed begin
    xh,gh,μ = collect_solutions(sol)
  end
  uh = Broadcasting(x->getindex(x,1:Ns[1],:))(xh)
  ph = Broadcasting(x->getindex(x,Ns[1]+1:Ns[1]+Ns[2],:))(xh)
  usnap,psnap,gsnap = Snapshots(:u,uh),Snapshots(:p,ph),Snapshots(:g,gh)
  save.((fepath,fepath,fepath,fepath),(usnap,psnap,gsnap,μ))
  save(fepath,Dict("FE time"=>time))
  usnap,psnap,μ,gsnap
end

function collect_solutions(sol)
  solk(k::Int) = collect_solutions(sol[k],k)
  results = solk.(eachindex(sol))
  xh = Matrix.(Broadcasting(r->getindex(r,1))(results))
  gh = Matrix.(Broadcasting(r->getindex(r,2))(results))
  μ = Broadcasting(r->getindex(r,3))(results)
  xh,gh,μ
end

function collect_solutions(solk,k::Int)
  println("\n Collecting solution $k")
  xh,gh = Vector{Float}[],Vector{Float}[]
  collect_solution!(xh,gh,solk)
  xh,gh,solk.psol.μ
end

function collect_solution!(
  x::Vector{Vector{Float}},
  g::Vector{Vector{Float}},
  solk::ParamFESolution)

  μk = solk.psol.μ

  xk = get_free_dof_values(solk.psol.uh)
  gk = get_dirichlet_dof_values(solk.trial(μk))
  push!(x,copy(xk))
  push!(g,copy(gk))
  x,g
end

function collect_solution!(
  x::Vector{Vector{Float}},
  g::Vector{Vector{Float}},
  solk::ParamTransientFESolution)

  μk = solk.psol.μ
  dt = solk.psol.solver.dt
  θ = solk.psol.solver.θ

  k = 1
  for (xk,t) in solk
    println("Time step: $k")
    tθ = t-dt*(1-θ)
    push!(x,copy(xk))
    push!(g,copy(get_dirichlet_dof_values(solk.trial(μk,tθ))))
    k += 1
  end
  x,g
end
