"""Create a directory at the given path"""
function create_dir(path::String)
  if !isdir(path)
    mkdir(path)
  end
  return
end

"""Get a full list of subdirectories at a given root directory"""
function get_all_subdirectories(path::String)
  filter(isdir,readdir(path,join=true))
end

function correct_path(path::String)
  path[end-3:end] == ".csv" ? path : path * ".csv"
end

function load_CSV(::Vector{Matrix{T}}, path::String) where T
  Array(CSV.read(correct_path(path), DataFrame))
end

function load_CSV(::Vector{Vector{T}}, path::String) where T
  var = Array(CSV.read(correct_path(path), DataFrame))
  try
    matrix_to_vecblocks(var)
  catch
    [parse.(T, split(chop(var[k]; head=1, tail=1), ",")) for k in 1:size(var)[1]]
  end
end

function load_CSV(::Array{T,D}, path::String) where {T,D}
  if D == 1
    Matrix{T}(CSV.read(correct_path(path), DataFrame))[:]
  else
    Array{T,D}(CSV.read(correct_path(path), DataFrame))
  end
end

function load_CSV(::SparseMatrixCSC{T}, path::String) where T
  var = Matrix{T}(CSV.read(correct_path(path), DataFrame))
  sparse(Int.(var[:,1]), Int.(var[:,2]), var[:,3])
end

function load_CSV(::SparseVector{T}, path::String) where T
  var = Matrix{T}(CSV.read(correct_path(path), DataFrame))
  sparse(Int.(var[:,1]), var[:,2])
end

function save_CSV(var::Array{T,D}, path::String) where {T,D}
  if D == 1
    var = reshape(var, :, 1)
  end
  try
    CSV.write(correct_path(path), DataFrame(var, :auto))
  catch
    CSV.write(correct_path(path), Tables.table(var))
  end
  return
end

function save_CSV(var::Vector{<:AbstractArray{T}}, path::String) where T
  save_CSV(blocks_to_matrix(var), correct_path(path))
end

function save_CSV(var::SparseMatrixCSC{T}, path::String) where T
  i, j, v = findnz(var)::Tuple{Vector{Int},Vector{Int},Vector{T}}
  CSV.write(correct_path(path), DataFrame([:i => i, :j => j, :v => v]))
  return
end

function save_CSV(var::SparseVector{T}, path::String) where T
  i, v = findnz(var)::Tuple{Vector{Int},Vector{T}}
  CSV.write(correct_path(path), DataFrame([:i => i, :v => v]))
  return
end

function save_CSV(var::Vector{<:AbstractArray{T}}, path::Vector{String}) where T
  @assert length(var) == length(path) "length(var) not equals to length(path)"
  Broadcasting(save_CSV)(var, path)
  return
end

function append_CSV(var::AbstractArray, path::String)
  pathCSV = correct_path(path)
  if !isfile(pathCSV)
    save_CSV(var, pathCSV)
  else
    file = open(pathCSV)
    save_CSV(var, file)
    close(file)
  end
  return
end

function save_structures_in_list(
  list_structures::Tuple,
  list_names::NTuple{D},
  path::String) where D

  @assert length(list_structures) == D "Wrong length"

  l_info_vec = [[l_idx,l_val] for (l_idx,l_val) in
    enumerate(list_structures) if !all(isempty.(l_val))]

  if !isempty(l_info_vec)
    l_info_mat = reduce(vcat,transpose.(l_info_vec))
    l_idx,l_val = l_info_mat[:,1], transpose.(l_info_mat[:,2])
    for (i₁,i₂) in enumerate(l_idx)
      save_CSV(l_val[i₁], joinpath(path, list_names[i₂]*".csv"))
    end
  end

  return

end

function load_structures_in_list(
  list_names::Tuple{Vararg{String, D}},
  list_types::Tuple,
  path::String) where D

  @assert length(list_types) == D "Wrong length"

  ret_tuple = ()

  for (idx, name) in enumerate(list_names)
    ret_tuple = (ret_tuple...,
      load_CSV(list_types[idx], joinpath(path, name*".csv")))
  end

  ret_tuple

end

function generate_dcube_model(
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
    domain = (0, 1, 0, 1, 0, 1)
    partition = (npart,npart,npart)
  end
  model = CartesianDiscreteModel(domain,partition)
  to_json_file(model,path)
  return
end
