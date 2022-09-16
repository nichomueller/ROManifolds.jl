function FEM_paths(root, problem_steadiness, problem_name, mesh_name, case)

  @assert isdir(root) "$root is an invalid root directory"

  root_tests = joinpath(root, "tests")
  create_dir(root_tests)
  mesh_path = joinpath(root_tests, joinpath("meshes", mesh_name))
  @assert isfile(mesh_path) "$mesh_path is an invalid mesh path"
  type_path = joinpath(root_tests, problem_steadiness)
  create_dir(type_path)
  problem_path = joinpath(type_path, problem_name)
  create_dir(problem_path)
  problem_and_info_path = joinpath(problem_path, "case" * string(case))
  create_dir(problem_and_info_path)
  current_test = joinpath(problem_and_info_path, mesh_name)
  create_dir(current_test)
  FEM_path = joinpath(current_test, "FEM_data")
  create_dir(FEM_path)
  FEM_snap_path = joinpath(FEM_path, "snapshots")
  create_dir(FEM_snap_path)
  FEM_structures_path = joinpath(FEM_path, "FEM_structures")
  create_dir(FEM_structures_path)

  FEMPathInfo(mesh_path, current_test, FEM_snap_path, FEM_structures_path)

end

function get_problem_id(problem_name::String)
  if problem_name == "poisson"
    return (0,)
  elseif problem_name == "ADR"
    return (0,0)
  elseif problem_name == "stokes"
    return (0,0,0)
  elseif problem_name == "navier-stokes"
    return (0,0,0,0)
  else
    error("unimplemented")
  end
end

function init_FEM_variables()

  M = sparse([], [], Float[])
  A = sparse([], [], Float[])
  B = sparse([], [], Float[])
  Xᵘ₀ = sparse([], [], Float[])
  Xᵘ = sparse([], [], Float[])
  Xᵖ₀ = sparse([], [], Float[])
  F = Vector{Float}(undef,0)
  H = Vector{Float}(undef,0)

  M, A, B, Xᵘ₀, Xᵘ, Xᵖ₀, F, H

end

function nonlinearity_lifting_op(FEMInfo::Info)
  if "A" ∉ FEMInfo.probl_nl && "L" ∉ FEMInfo.probl_nl
    return 0
  elseif "A" ∈ FEMInfo.probl_nl && "L" ∉ FEMInfo.probl_nl
    return 1
  elseif "A" ∉ FEMInfo.probl_nl && "L" ∈ FEMInfo.probl_nl
    return 2
  else
    return 3
  end
end

function get_FEMProblem_info(FEMInfo::Info)
  μ = load_CSV(Array{Float}[],
    joinpath(FEMInfo.Paths.FEM_snap_path, "μ.csv"))::Vector{Vector{Float}}
  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)
  FEMSpace = get_FEMSpace₀(FEMInfo.problem_id, FEMInfo,model)

  FEMSpace, μ

end

function get_h(FEMSpace::Problem)
  Λ = SkeletonTriangulation(FEMSpace.Ω)
  dΛ = Measure(Λ, 2)
  h = get_array(∫(1)dΛ)[1]
  h
end

function get_α_stab(
  FEMSpace::FEMProblemS,
  Param::ParamInfoS)

  h_mesh = get_h(FEMSpace)
  Pechlet(x) = norm(Param.b(x))*h_mesh / (2*Param.α(x))
  ξ(x) = x - 1 + 2*x/(exp(2*x)-1)
  α_stab(x) = Param.α(x)*(1 + ξ(Pechlet(x)))

  α_stab

end

function get_α_stab(
  FEMSpace::FEMProblemST,
  Param::ParamInfoST)

  h_mesh = get_h(FEMSpace)
  Pechlet(x, t) = norm(Param.b(x, t))*h_mesh / (2*Param.α(x, t))
  ξ(x) = x - 1 + 2*x/(exp(2*x)-1)
  α_stab(x, t::Real) = Param.α(x, t)*(1 + ξ(Pechlet(x, t)))
  α_stab(t::Real) = x -> α_stab(x, t)

  α_stab

end

function get_timesθ(FEMInfo::FEMInfoST)
  collect(FEMInfo.t₀:FEMInfo.δt:FEMInfo.tₗ-FEMInfo.δt).+FEMInfo.δt*FEMInfo.θ
end

function generate_vtk_file(
  FEMSpace::FEMProblem,
  path::String,
  var_name::String,
  var::Array)

  FE_var = FEFunction(FEMSpace.V, var)
  writevtk(FEMSpace.Ω, path, cellfields = [var_name => FE_var])

end

function find_FE_elements(
  V₀::UnconstrainedFESpace,
  trian::BodyFittedTriangulation,
  idx::Vector)

  connectivity = get_cell_dof_ids(V₀, trian)::Table{Int32, Vector{Int32}, Vector{Int32}}

  el = Int[]
  for i = 1:length(idx)
    for j = 1:size(connectivity)[1]
      if idx[i] in abs.(connectivity[j])
        append!(el, Int(j))
      end
    end
  end

  unique(el)

end

function find_FE_elements(
  V₀::UnconstrainedFESpace,
  trian::BoundaryTriangulation,
  idx::Vector)

  connectivity = collect(get_cell_dof_ids(V₀, trian))::Vector{Vector{Int32}}

  el = Int[]
  for i = 1:length(idx)
    for j = 1:size(connectivity)[1]
      if idx[i] in abs.(connectivity[j])
        append!(el, Int(j))
      end
    end
  end

  unique(el)

end

function define_g_FEM(
  FEMSpace::FEMProblemS,
  Param::ParamInfoS)

  interpolate_dirichlet(Param.g, FEMSpace.V)

end

function define_dg_FEM(
  FEMSpace::FEMProblemST,
  Param::ParamInfoST)

  function dg(t)
    dg(x,t::Real) = ∂t(Param.g)(x,t)
    dg(t::Real) = x -> dg(x,t)
    interpolate_dirichlet(dg(t), FEMSpace.V(t))
  end

end

function set_labels(
  model::DiscreteModel,
  bnd_info::Dict)

  tags = collect(keys(bnd_info))
  bnds = collect(values(bnd_info))
  @assert length(tags) == length(bnds)

  labels = get_face_labeling(model)
  for i = eachindex(tags)
    if tags[i] ∉ labels.tag_to_name
      add_tag_from_tags!(labels, tags[i], bnds[i])
    end
  end

end

function generate_dcube_discrete_model(
  FEMInfo::Info,
  d::Int,
  npart::Int,
  mesh_name::String)

  if !occursin(".json",mesh_name)
    mesh_name *= ".json"
  end
  mesh_dir = FEMInfo.Paths.mesh_path[1:findall(x->x=='/',FEMInfo.Paths.mesh_path)[end]]
  mesh_path = joinpath(mesh_dir,mesh_name)
  generate_dcube_discrete_model(d, npart, mesh_path)

end

function generate_dcube_discrete_model(
  d::Int,
  npart::Int,
  path::String)

  @assert d ≤ 3 "Select d-dimensional domain, where d ≤ 3"
  if d == 1
    domain = (0,1)
    partition = (npart)
  elseif d == 2
    domain = (0,1,0,1)
    partition = (npart,npart)
  else
    domain = (0,1,0,1,0,1)
    partition = (npart,npart,npart)
  end
  model = CartesianDiscreteModel(domain,partition)
  to_json_file(model,path)
end
