function FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)

  @assert isdir(root) "$root is an invalid root directory"

  nonlins = ""
  for (key, value) in problem_nonlinearities
      if value === true
          nonlins *= "_" * key
      end
  end

  root_tests = joinpath(root, "tests")
  create_dir(root_tests)
  mesh_path = joinpath(root_tests, joinpath("meshes", mesh_name))
  @assert isfile(mesh_path) "$mesh_path is an invalid mesh path"
  type_path = joinpath(root_tests, problem_type)
  create_dir(type_path)
  problem_path = joinpath(type_path, problem_name)
  create_dir(problem_path)
  problem_and_info_path = joinpath(problem_path, string(problem_dim) * "D" * nonlins)
  create_dir(problem_and_info_path)
  current_test = joinpath(problem_and_info_path, mesh_name)
  create_dir(current_test)
  FEM_path = joinpath(current_test, "FEM_data")
  create_dir(FEM_path)
  FEM_snap_path = joinpath(FEM_path, "snapshots")
  create_dir(FEM_snap_path)
  FEM_structures_path = joinpath(FEM_path, "FEM_structures")
  create_dir(FEM_structures_path)

  _ -> (mesh_path; current_test; FEM_snap_path; FEM_structures_path)

end

function stretching(x::Point, μ::Float64)
    #=MODIFY
    =#

    m = zeros(length(x))
    m[1] = μ * x[1]^2
    for i in 2:length(x)
        m[i] = μ * x[i]
    end

    Point(m)

end

struct reference_info{T<:Int64}
  L::T
  dim::T
  ndof_dir::T
end

function generate_cartesian_model(info::reference_info, deformation::Function, μ::Float64)
    #=MODIFY
    =#
  pmin = Point(Fill(0, info.dim))
  pmax = Point(Fill(info.L, info.dim))
  partition = Tuple(Fill(info.ndof_dir, info.dim))

  model = CartesianDiscreteModel(pmin, pmax, partition, map = (x->deformation(x, μ)))

  return model

end

function generate_vtk_file(FE_space::FEMProblem, path::String, var_name::String, var::Array)

  FE_var = FEFunction(FE_space.V, var)
  writevtk(FE_space.Ω, path, cellfields = [var_name => FE_var])

end
