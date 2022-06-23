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

function load_CSV(path::String) where T
  Matrix{Float64}(CSV.read(path, DataFrame))
end

function load_CSV(path::String;::SparseMatrixCSC{T}) where T
  var = Matrix{Float64}(CSV.read(path, DataFrame))
  sparse(convert(Vector{Int64}, var[:,1]), convert(Vector{Int64}, var[:,2]), var[:,3])
end

function save_CSV(var::AbstractArray, file_name::String)
  if length(size(var)) == 1
    var = reshape(var, :, 1)
  end
  if issparse(var)
    i, j, v = findnz(var)
    CSV.write(file_name, DataFrame([:i => i, :j => j, :v => v]))
  else
    try
      CSV.write(file_name, DataFrame(var, :auto))
    catch
      CSV.write(file_name, Tables.table(var))
    end
  end
end

function append_CSV(var::AbstractArray, file_name::String)
  if !isfile(file_name)
    save_CSV(var, file_name)
  else
    file = open(file_name)
    save_CSV(var, file)
    close(file)
  end
end
