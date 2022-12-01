include("../FEM/FEM.jl")
include("FunctionDefinitions.jl")

test_path(root,mesh,::Val{true}) = joinpath(root,"steady/$mesh")
test_path(root,mesh,::Val{false}) = joinpath(root,"unsteady/$mesh")

function fem_path(
  ptype::ProblemType,
  mesh::String,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes")

  @assert isdir(root) "Provide valid root path"
  tpath = test_path(root,mesh,issteady(ptype))
  fepath = joinpath(tpath,"fem")
  create_dir!(fepath)
  fepath
end

function rom_path(
  ptype::ProblemType,
  mesh::String,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes")

  @assert isdir(root) "Provide valid root path"
  tpath = test_path(root,mesh,issteady(ptype))
  rbpath = joinpath(tpath,"rom")
  create_dir!(rbpath)
  rbpath
end

function rom_offline_path(
  mesh::String,
  ptype::ProblemType,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes")

  rb_off_path = joinpath(rom_path(ptype,mesh,root),"offline")
  create_dir!(rb_off_path)
  rb_off_path
end

function rom_online_path(
  ptype::ProblemType,
  mesh::String,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes")

  rb_on_path = joinpath(rom_path(ptype,mesh,root),"online")
  create_dir!(rb_on_path)
  rb_on_path
end

function rom_off_on_paths(
  ptype::ProblemType,
  mesh::String,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes")
  rom_offline_path(ptype,mesh,root)
end

function mesh_path(
  mesh::String,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes")

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
  bnd_info::Dict,
  ptype::ProblemType)

  mshpath = mesh_path(mesh,root)
  model_info(mshpath,bnd_info,ispdomain(ptype))
end

function get_fe_snapshots(ptype::ProblemType,solver,op,fepath::String,run_fe::Bool,args...)
  if run_fe
    get_fe_snapshots(ptype,solver,op,fepath,args...)
  else
    get_fe_snapshots(ptype,fepath)
  end
end

get_fe_snapshots(ptype::ProblemType,fepath::String) =
  get_fe_snapshots(isindef(ptype),fepath)

function get_fe_snapshots(::Val{false},fepath::String)
  uh,μ = load.(joinpath(fepath,"uh"),joinpath(fepath,"μ"))
  Snapshots.([:u,:μ],[uh,μ])
end

function get_fe_snapshots(::Val{true},fepath::String)
  uh,ph,μ = load.(joinpath(fepath,"uh"),joinpath(fepath,"ph"),joinpath(fepath,"μ"))
  Snapshots.([:u,:p,:μ],[uh,ph,μ])
end

function get_fe_snapshots(
  ptype::ProblemType,
  solver::FESolver,
  op::ParamFEOperator,
  fepath::String,
  n=100)

  sol = solve(solver,op,n)
  get_fe_snapshots(isindef(ptype),sol,fepath)
end

function get_fe_snapshots(
  ptype::ProblemType,
  solver::ThetaMethod,
  op::ParamTransientFEOperator,
  fepath::String,
  t0::Real,
  tF::Real,
  n=100)

  sol = solve(solver,op,t0,tF,n)
  get_fe_snapshots(isindef(ptype),sol,fepath)
end

function get_fe_snapshots(::Val{false},sol,fepath::String)
  uh,μ = collect_solutions(sol)
  usnap,μsnap = Snapshots.([:u,:μ],[uh,μ])
  save.([fepath,fepath],[usnap,μsnap])
  usnap,μsnap
end

function get_fe_snapshots(::Val{false},sol,fepath::String)
  Ns = get_Ns(op)
  uh,μ = collect_solutions(sol)
  uh,ph = uh[1:Ns[1],:],uh[Ns[1]+1:end,:]
  usnap,psnap,μsnap = Snapshots.([:u,:p,:μ],[uh,ph,μ])
  save.([fepath,fepath,fepath],[usnap,psnap,μsnap])
  usnap,psnap,μsnap
end

function collect_solutions(sol)
  solk(k::Int) = collect_solutions(sol[k],k)
  results = solk.(eachindex(sol))
  Matrix(first.(results)),last.(results)
end

function collect_solutions(solk::ParamFESolution,k::Int)
  println("\n Collecting solution $k")
  solk.uh,solk.μ
end

function collect_solutions(solk::ParamTransientFESolution,k::Int)
  println("\n Collecting solution $k")
  uh = allocate_vector(Float)
  collect_solutions!(uh,solk)
  uh,solk.odesol.μ
end

function collect_solutions!(
  x::Vector{Vector{Float}},
  solk::ParamTransientFESolution)

  k = 1
  for (uh,_) in solk
    println("Time step: $k")
    push!(x,get_free_dof_values(uh))
    k += 1
  end
  Matrix(x)
end
