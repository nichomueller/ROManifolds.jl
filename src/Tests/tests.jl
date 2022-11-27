include("../FEM/FEM.jl")

function fem_path(
  mesh::String,
  S::Bool;
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes")

  @assert isdir(root) "Provide valid root path"
  test_path = S ? joinpath(root,"steady/$mesh") : joinpath(root,"unsteady/$mesh")
  fepath = joinpath(test_path,"fem")
  create_dir!(fepath)
  fepath
end

function configure(
  degree::Int,
  ranges::Vector{Vector{Float}};
  root="/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes",
  mesh="cube5x5x5.json",
  bnd_info=Dict("dirichlet" => collect(1:25),"neumann" => [26]),
  sampling=UniformSampling(),
  S::Bool)

  function set_labels!()
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

  fepath = fem_path(mesh,S;root)
  mesh_path = joinpath(get_parent_dir(root),"meshes/$mesh")
  model = DiscreteModelFromFile(mesh_path)
  set_labels!()

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  PS = ParamSpace(ranges,sampling)

  fepath,model,dΩ,dΓn,PS
end

function run(
  solver::FESolver,
  op::ParamFEOperator,
  n=100)

  sol = solve(solver,op,n)
  uh,μ = collect_solutions(sol)
  Snapshot(:u,uh),Snapshot(:μ,μ)
end

function run(
  solver::ThetaMethod,
  op::ParamTransientFEOperator,
  t0::Real,
  tF::Real,
  n=100)

  sol = solve(solver,op,t0,tF,n)
  uh,μ = collect_solutions(sol)
  Snapshot(:u,uh),Snapshot(:μ,μ)
end

function collect_solutions(sol)
  solk(k::Int) = collect_solutions(sol[k],k)
  results = Broadcasting(solk)(eachindex(sol))
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
