include("../FEM/FEM.jl")
include("FunctionDefinitions.jl")

test_path(root,mesh,::Val{true}) = joinpath(root,"steady/$mesh")
test_path(root,mesh,::Val{false}) = joinpath(root,"unsteady/$mesh")

function fem_path(
  mesh::String,
  ptype::ProblemType,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes")

  @assert isdir(root) "Provide valid root path"
  tpath = test_path(root,mesh,issteady(ptype))
  fepath = joinpath(tpath,"fem")
  create_dir!(fepath)
  fepath
end

function rom_path(
  mesh::String,
  ptype::ProblemType,
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

  rb_off_path = joinpath(rom_path(mesh,ptype,root),"offline")
  create_dir!(rb_off_path)
  rb_off_path
end

function rom_online_path(
  mesh::String,
  ptype::ProblemType,
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes")

  rb_on_path = joinpath(rom_path(mesh,ptype,root),"online")
  create_dir!(rb_on_path)
  rb_on_path
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
  degree::Int,
  ::Val{false})

  model = DiscreteModelFromFile(mshpath)
  set_labels!(model,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  model,dΩ,dΓn
end

function model_info(
  ::String,
  bnd_info::Dict,
  degree::Int,
  ::Val{false})

  function model(μ) end
  set_labels!(model,bnd_info)
  Ω(μ) = Triangulation(model(μ))
  dΩ(μ) = Measure(Ω(μ),degree)
  Γn(μ) = BoundaryTriangulation(model(μ),tags=["neumann"])
  dΓn(μ) = Measure(Γn(μ),degree)

  model,dΩ,dΓn
end

function model_info(
  bnd_info::Dict,
  degree::Int,
  ptype::ProblemType)

  mshpath = mesh_path(mesh,root)
  model_info(mshpath,bnd_info,degree,ispdomain(ptype))
end

function get_fe_snapshots(solver,op,fepath::String,run_fe::Bool,args...)
  run_fe ? get_fe_snapshots(solver,op,fepath,args...) : get_fe_snapshots(fepath)
end

function get_fe_snapshots(fepath::String)
  uh,μ = allocate_snapshot.([:u,:μ],[Matrix{Float},Vector{Param}])
  load!(uh,fepath),load!(μ,fepath)
end

function get_fe_snapshots(
  solver::FESolver,
  op::ParamFEOperator,
  fepath::String,
  n=100)

  sol = solve(solver,op,n)
  uh,μ = collect_solutions(sol)
  usnap,μsnap = Snapshot(:u,uh),Snapshot(:μ,μ)
  save(usnap,fepath),save(μsnap,fepath)
  usnap,μsnap
end

function get_fe_snapshots(
  solver::ThetaMethod,
  op::ParamTransientFEOperator,
  fepath::String,
  t0::Real,
  tF::Real,
  n=100)

  sol = solve(solver,op,t0,tF,n)
  uh,μ = collect_solutions(sol)
  usnap,μsnap = Snapshot(:u,uh),Snapshot(:μ,μ)
  save(usnap,fepath),save(μsnap,fepath)
  usnap,μsnap
end

function collect_solutions(sol)
  solk(k::Int) = collect_solutions(sol[k],k)
  results = solk.(eachindex(sol))
  blocks_to_matrix(first.(results)),last.(results)
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
  blocks_to_matrix(x)
end
