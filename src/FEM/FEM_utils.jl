function FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, case)

  @assert isdir(root) "$root is an invalid root directory"

  root_tests = joinpath(root, "tests")
  create_dir(root_tests)
  mesh_path = joinpath(root_tests, joinpath("meshes", mesh_name))
  @assert isfile(mesh_path) "$mesh_path is an invalid mesh path"
  type_path = joinpath(root_tests, problem_type)
  create_dir(type_path)
  problem_path = joinpath(type_path, problem_name)
  create_dir(problem_path)
  problem_and_info_path = joinpath(problem_path, string(problem_dim) * "D" * "_" * string(case))
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

function find_FE_elements(problem_info, ROM_info, idx::Array)

  parametric_info = get_parametric_specifics(ROM_info, [])
  FE_space = get_FE_space(problem_info, parametric_info.model)

  el = Int64[]
  for i = 1:length(idx)
    for j = 1:size(FE_space.σₖ)[1]
      if idx[i] in abs.(FE_space.σₖ[j])
        append!(el, j)
      end
    end
  end

  unique(el)

end

function from_vec_to_mat_idx(idx::Array, Nᵤ::Int64)

  row_idx = Int.(idx .% Nᵤ)
  row_idx[findall(x->x===0, row_idx)] .= Nᵤ
  col_idx = Int.((idx-row_idx)/Nᵤ .+ 1)

  row_idx, col_idx

end

function from_spacetime_to_space_time_idx_mat(idx::Array, Nᵤ::Int64)

  idx_time = 1 .+ floor.((idx.-1)/Nᵤ^2)
  idx_space = idx - (idx_time.-1)*Nᵤ^2

  idx_space, idx_time

end

function from_spacetime_to_space_time_idx_vec(idx::Array, Nᵤ::Int64)

  idx_time = 1 .+ floor.((idx.-1)/Nᵤ)
  idx_space = idx - (idx_time.+1)*Nᵤ

  idx_space, idx_time

end

function interface_to_unit_circle(map::Function, pts::Array)

  return map.(pts)

end

function chebyshev_polynomial(x::Float64, n::Int64)

  if n === 0
    return 1
  elseif n === 1
    return 2*x
  else
    return 2*x*chebyshev_polynomial(x,n-1) - chebyshev_polynomial(x,n-2)
  end

end

function chebyschev_multipliers(x::Array, order::Int64, dim=3)

  Ξ = Matrix{Float64}[]
  for d = 1:dim
    for n = 1:order
      for k = 1:n
        ωₖ = k*pi/(order+1)
        Pⁿₖ = chebyshev_polynomial(x[1]*cos(ωₖ*x[1]) + x[2]*sin(ωₖ*x[2]), n)/sqrt(pi)
        append!(Ξ, Pⁿₖ*I(dim)[:,d])
      end
    end
  end

  return Ξ

end
