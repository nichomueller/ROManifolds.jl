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
    var = Array(CSV.read(path, DataFrame))
    return [parse.(T, split(chop(var[k]; head=1, tail=1), ',')) for k in 1:size(var)[1]]
  catch
    var = CSV.read(path, DataFrame)
    return [var[:,i] for i in 1:size(var,2)]
  end
end

function load_CSV(::Array{D,T}, path::String) where {D,T}
  return Array{D,T}(CSV.read(path, DataFrame))
end

function load_CSV(::SparseMatrixCSC{T}, path::String) where T
  var = Matrix{T}(CSV.read(path, DataFrame))
  sparse(Int.(var[:,1]), Int.(var[:,2]), var[:,3])
end

function load_CSV(::SparseVector{T}, path::String) where T
  var = Matrix{T}(CSV.read(path, DataFrame))
  sparse(Int.(var[:,1]), var[:,2])
end

function save_CSV(var::Array{D,T}, path::String) where {D,T}

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
  list_names::Tuple,
  path::String)

  @assert length(list_structures) == length(list_names) "Wrong dimension"

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
