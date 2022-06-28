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

function load_CSV(::Array{N,T}, path::String) where {N,T}
  return Array{N,T}(CSV.read(path, DataFrame))
end

function load_CSV(::SparseMatrixCSC{T}, path::String) where T
  var = Matrix{T}(CSV.read(path, DataFrame))
  sparse(Int.(var[:,1]), Int.(var[:,2]), var[:,3])
end

function load_CSV(::SparseVector{T}, path::String) where T
  var = Matrix{T}(CSV.read(path, DataFrame))
  sparse(Int.(var[:,1]), var[:,2])
end

function save_CSV(var::Array{N,T}, path::String) where {N,T}

  if N == 1
    var = reshape(var, :, 1)
  end

  try
    CSV.write(path, DataFrame(var, :auto))
  catch
    CSV.write(path, Tables.table(var))
  end

end

function save_CSV(var::SparseMatrixCSC{T}, path::String) where T
  i, j, v = findnz(var)
  CSV.write(path, DataFrame([:i => i, :j => j, :v => v]))
end

function save_CSV(var::SparseVector{T}, path::String) where T
  i, v = findnz(var)
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
