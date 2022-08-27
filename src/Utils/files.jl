"""Create a directory at the given path"""
function create_dir(path::String)
  if !isdir(path)
    mkdir(path)
  end
end

"""Get a full list of subdirectories at a given root directory"""
function get_all_subdirectories(path::String)
  filter(isdir,readdir(path,join=true))
end

function load_CSV(::Array{Array{T}}, path::String) where T
  try
    var = CSV.read(path, DataFrame)
    return [T.(var[:,i]) for i in 1:size(var,2)]
  catch
    var = Array(CSV.read(path, DataFrame))
    return [parse.(T, split(chop(var[k]; head=1, tail=1), ',')) for k in 1:size(var)[1]]
  end
end

function load_CSV(::Array{T,D}, path::String) where {T,D}
  if D == 1
    return Matrix{T}(CSV.read(path, DataFrame))[:]
  else
    return Array{T,D}(CSV.read(path, DataFrame))
  end
end

function load_CSV(::SparseMatrixCSC{T}, path::String) where T
  var = Matrix{T}(CSV.read(path, DataFrame))
  sparse(Int.(var[:,1]), Int.(var[:,2]), var[:,3])
end

function load_CSV(::SparseVector{T}, path::String) where T
  var = Matrix{T}(CSV.read(path, DataFrame))
  sparse(Int.(var[:,1]), var[:,2])
end

function save_CSV(var::Array{T,D}, path::String) where {T,D}

  if D == 1
    var = reshape(var, :, 1)
  end

  try
    CSV.write(path, DataFrame(var, :auto))
  catch
    CSV.write(path, Tables.table(var))
  end

end

function save_CSV(var::SparseMatrixCSC{T}, path::String) where T
  i, j, v = findnz(var)::Tuple{Vector{Int},Vector{Int},Vector{T}}
  CSV.write(path, DataFrame([:i => i, :j => j, :v => v]))
end

function save_CSV(var::SparseVector{T}, path::String) where T
  i, v = findnz(var)::Tuple{Vector{Int},Vector{T}}
  CSV.write(path, DataFrame([:i => i, :v => v]))
end

function append_CSV(var::AbstractArray, path::String)
  if !isfile(path)
    save_CSV(var, path)
  else
    file = open(path)
    save_CSV(var, file)
    close(file)
  end
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
