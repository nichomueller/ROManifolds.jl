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

  _ -> (mesh_path, current_test, FEM_snap_path, FEM_structures_path)

end

function get_problem_id(problem_name::String)
  if problem_name == "poisson"
    return (0,)
  elseif problem_name == "stokes"
    return (0,0)
  elseif problem_name == "navier-stokes"
    return (0,0,0)
  else
    error("unimplemented")
  end
end

function init_FEM_variables()

  M = sparse([], [], Float64[])
  A = sparse([], [], Float64[])
  B = sparse([], [], Float64[])
  Xᵘ₀ = sparse([], [], Float64[])
  Xᵘ = sparse([], [], Float64[])
  Xᵖ₀ = sparse([], [], Float64[])
  F = Vector{Float64}(undef,0)
  H = Vector{Float64}(undef,0)

  M, A, B, Xᵘ₀, Xᵘ, Xᵖ₀, F, H

end

function nonlinearity_lifting_op(FEMInfo::Info)
  if !FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
    return 0
  elseif FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
    return 1
  elseif !FEMInfo.probl_nl["A"] && FEMInfo.probl_nl["g"]
    return 2
  else
    return 3
  end
end

function get_timesθ(FEMInfo::UnsteadyInfo)
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

  el = Int64[]
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

  el = Int64[]
  for i = 1:length(idx)
    for j = 1:size(connectivity)[1]
      if idx[i] in abs.(connectivity[j])
        append!(el, Int(j))
      end
    end
  end

  unique(el)

end

function set_labels(FEMInfo::Info, model::DiscreteModel)

  labels = get_face_labeling(model)
  if !isempty(FEMInfo.dirichlet_tags) && !isempty(FEMInfo.dirichlet_bnds)
    for i = eachindex(FEMInfo.dirichlet_tags)
      if FEMInfo.dirichlet_tags[i] ∉ labels.tag_to_name
        add_tag_from_tags!(labels, FEMInfo.dirichlet_tags[i], FEMInfo.dirichlet_bnds[i])
      end
    end
  end
  if !isempty(FEMInfo.neumann_tags) && !isempty(FEMInfo.neumann_bnds)
    for i = eachindex(FEMInfo.neumann_tags)
      if FEMInfo.neumann_tags[i] ∉ labels.tag_to_name
        add_tag_from_tags!(labels, FEMInfo.neumann_tags[i], FEMInfo.neumann_bnds[i])
      end
    end
  end

  labels

end

function generate_dcube_discrete_model(
  FEMInfo::Info,
  d::Int,
  npart::Int,
  mesh_name::String)

  if !occursin(".json",mesh_name)
    mesh_name *= ".json"
  end
  mesh_dir = FEMInfo.paths.mesh_path[1:findall(x->x=='/',FEMInfo.paths.mesh_path)[end]]
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
