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

function generate_vtk_file(FE_space::FEMProblem, path::String, var_name::String, var::Array)

  FE_var = FEFunction(FE_space.V, var)
  writevtk(FE_space.Ω, path, cellfields = [var_name => FE_var])

end

function find_FE_elements(V₀::Gridap.FESpaces.UnconstrainedFESpace, trian::Triangulation, idx::Array)

  connectivity = get_cell_dof_ids(V₀, trian)

  el = Int64[]
  for i = 1:length(idx)
    for j = 1:size(connectivity)[1]
      if idx[i] in abs.(connectivity[j])
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

  idx_time = 1 .+ floor.(Int64,(idx.-1)/Nᵤ^2)
  idx_space = idx - (idx_time.-1)*Nᵤ^2

  idx_space, idx_time

end

function from_spacetime_to_space_time_idx_vec(idx::Array, Nᵤ::Int64)

  idx_time = 1 .+ floor.(Int64,(idx.-1)/Nᵤ)
  idx_space = idx - (idx_time.-1)*Nᵤ

  idx_space, idx_time

end

function from_full_idx_to_sparse_idx(sparse_to_full_idx::Vector,full_idx::Vector,Nₛ::Int64)

  #sparse_idx = zeros(Int64,length(full_idx)*Nₜ)
  #[sparse_idx[j+(i-1)*length(row_idx)] = row_idx[j] + (i-1)*Nₛ^2 for i=1:Nₜ for j=1:length(full_idx)]
  #return sparse_idx
  Nfull  = length(sparse_to_full_idx)
  full_idx_space,full_idx_time = from_spacetime_to_space_time_idx_vec(full_idx, Nfull)
  sparse_idx = (full_idx_time.-1)*Nₛ^2+row_idx[full_idx_space]
  return sparse_idx

end

function remove_zero_entries(M_sparse::SparseMatrixCSC) :: Matrix
  for col = 1:size(M_sparse)[2]
    _,vals = findnz(M_sparse[:,col])
    if col == 1
      global M = zeros(length(vals),size(M_sparse)[2])
    end
    M[:,col] = vals
  end
  return M
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
