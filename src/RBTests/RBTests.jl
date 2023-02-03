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
  rompath = joinpath(tpath,"rom")
  keytpath = joinpath(rompath,keyword())

  offpath = rom_offline_path(keytpath,ϵ)
  onpath = rom_online_path(keytpath,ϵ)
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
  args...;
  kwargs...)

  if run_fem
    printstyled("Generating $nsnap full order snapshots\n";color=:blue)
    generate_fe_snapshots(ptype,solver,op,fepath,nsnap,args...;kwargs...)
  else
    printstyled("Loading $nsnap full order snapshots\n";color=:blue)
    load_fe_snapshots(isindef(ptype),fepath,nsnap)
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
  nsnap::Int;
  kwargs...)

  sol = solve(solver,op,nsnap)
  generate_fe_snapshots(isindef(ptype),sol,fepath;kwargs...)
end

function generate_fe_snapshots(
  ptype::ProblemType,
  solver::ThetaMethod,
  op::ParamTransientFEOperator,
  fepath::String,
  nsnap::Int,
  t0::Real,
  tF::Real;
  kwargs...)

  sol = solve(solver,op,t0,tF,nsnap)
  generate_fe_snapshots(isindef(ptype),sol,fepath;kwargs...)
end

function generate_fe_snapshots(::Val{false},sol,fepath::String;save_snap=true)
  time = @elapsed begin
    uh,μ = collect_solutions(sol)
  end
  usnap = Snapshots(:u,uh)
  if save_snap
    save.((fepath,fepath),(usnap,μ))
    save(fepath,Dict("FE time"=>time))
  end
  usnap,μ
end

function generate_fe_snapshots(::Val{true},sol,fepath::String;save_snap=true)
  Ns = get_Ns(sol)
  time = @elapsed begin
    xh,μ = collect_solutions(sol)
  end
  uh = Broadcasting(x->getindex(x,1:Ns[1],:))(xh)
  ph = Broadcasting(x->getindex(x,Ns[1]+1:Ns[1]+Ns[2],:))(xh)
  usnap,psnap = Snapshots(:u,uh),Snapshots(:p,ph)
  if save_snap
    save.((fepath,fepath,fepath),(usnap,psnap,μ))
    save(fepath,Dict("FE time"=>time))
  end
  usnap,psnap,μ
end

function collect_solutions(sol)
  x,μ = allocate_solutions(first(sol))
  @threads for k in eachindex(sol)
    printstyled("Collecting solution $k\n";color=:blue)
    collect_solutions!(x,μ,sol[k])
  end
  Matrix.(x),μ
end

allocate_solutions(::ParamFESolution) = Vector{Float}[],Param[]

allocate_solutions(::ParamTransientFESolution) = Matrix{Float}[],Param[]

function collect_solutions!(x,μ,solk)
  collect_solution!(x,solk),push!(μ,solk.psol.μ)
end

function collect_solution!(
  x::Vector{Vector{Float}},
  solk::ParamFESolution)

  xk = get_free_dof_values(solk.psol.uh)
  push!(x,copy(xk))
  x,g
end

function collect_solution!(
  x::Vector{Matrix{Float}},
  solk::ParamTransientFESolution)

  n = 1
  xk = Vector{Float}[]
  for (xn,n) in solk
    printstyled("Time $n\n";color=:blue)
    push!(xk,copy(xn))
  end
  push!(x,Matrix(xk))
  x
end
