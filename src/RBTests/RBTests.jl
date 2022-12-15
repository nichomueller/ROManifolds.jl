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

function rom_path(ptype::ProblemType,mesh::String,root::String,ϵ::Float)
  @assert isdir(root) "Provide valid root path"
  tpath = test_path(root,mesh,issteady(ptype))
  rbpath = joinpath(joinpath(tpath,"rom"),"$ϵ")
  create_dir!(rbpath)
  rbpath
end

function rom_offline_path(ptype::ProblemType,mesh::String,root::String,ϵ::Float)
  rb_off_path = joinpath(rom_path(ptype,mesh,root,ϵ),"offline")
  create_dir!(rb_off_path)
  rb_off_path
end

function rom_online_path(ptype::ProblemType,mesh::String,root::String,ϵ::Float)
  rb_on_path = joinpath(rom_path(ptype,mesh,root,ϵ),"online")
  create_dir!(rb_on_path)
  rb_on_path
end

function rom_off_on_paths(ptype::ProblemType,mesh::String,root::String,ϵ::Float)
  offpath = rom_offline_path(ptype,mesh,root,ϵ)
  onpath = rom_online_path(ptype,mesh,root,ϵ)
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

function fe_snapshots(ptype::ProblemType,solver,op,fepath::String,run_fem::Bool,nsnap::Int,args...)
  if run_fem
    generate_fe_snapshots(ptype,solver,op,fepath,nsnap,args...)
  else
    load_fe_snapshots(ptype,fepath,nsnap)
  end
end

load_fe_snapshots(ptype::ProblemType,fepath::String,nsnap::Int) =
  load_fe_snapshots(isindef(ptype),fepath,nsnap)

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
  uh,μ = collect_solutions(sol)
  usnap = Snapshots(:u,uh)
  save(fepath,usnap)
  save(fepath,μ)
  usnap,μ
end

function generate_fe_snapshots(::Val{true},sol,fepath::String)
  Ns = get_Ns(sol)
  xh,μ = collect_solutions(sol)
  uh = Broadcasting(x->getindex(x,1:Ns[1],:))(xh)
  ph = Broadcasting(x->getindex(x,Ns[1]+1:Ns[1]+Ns[2],:))(xh)
  usnap,psnap = Snapshots.([:u,:p],[uh,ph])
  save.([fepath,fepath],[usnap,psnap])
  save(fepath,μ)
  usnap,psnap,μ
end

function collect_solutions(sol)
  solk(k::Int) = collect_solutions(sol[k],k)
  results = solk.(eachindex(sol))
  Matrix.(first.(results)),last.(results)
end

function collect_solutions(solk,k::Int)
  println("\n Collecting solution $k")
  uh = allocate_vblock(Float)
  collect_solutions!(uh,solk)
  uh,solk.psol.μ
end

function collect_solutions!(
  x::Vector{Vector{Float}},
  solk::ParamFESolution)

  uh = get_free_dof_values(solk.psol.uh)
  push!(x,copy(uh))
  x
end

function collect_solutions!(
  x::Vector{Vector{Float}},
  solk::ParamTransientFESolution)

  k = 1
  for (uh,_) in solk
    println("Time step: $k")
    push!(x,copy(uh))
    k += 1
  end
  x
end
