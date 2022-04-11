include("imports.jl")

function create_dir(path::String)
    #=Create a directory at the given path
    :param name: path of the directory to be created, if not already existing
    :type name: str
    =#

    if !isdir(path)
        mkdir(path)
    end

end

function get_full_subdirectories(rootdir::String)
    #=Get a full list of subdirectories at a given root directory
    :param rootdir: path of the root directory whose subdirectories are searched and listed
    :type rootdir: str
    =#

    return filter(isdir, readdir(rootdir, join = true))

end

function save_CSV(var::AbstractArray, file_name::String)

  if length(size(var)) === 1
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

function load_CSV(path::String; convert_to_sparse = false)

  var = Matrix(CSV.read(path, DataFrame))
  if convert_to_sparse === true
      var = sparse(convert(Vector{Int64}, var[:,1]), convert(Vector{Int64}, var[:,2]), var[:,3])
  end

  var

end


function mydot(vec1::Array, vec2::Array, norm_matrix = nothing)
  #=It computes the inner product between 'vec1' and 'vec2', defined by the (positive definite) matrix 'norm_matrix'.
  If 'norm_matrix' is None (default), the standard inner product between 'vec1' and 'vec2' is returned.
  :param vec1: first vector
  :type vec1: np.ndarray
  :param vec2: second vector
  :type vec2: np.ndarray
  :param norm_matrix: positive definite matrix, defining the inner product. If None, it defaults to the identity.
  :type norm_matrix: scipy.sparse.csc_matrix or np.ndarray or NoneType
  :return: inner product between 'vec1' and 'vec2', defined by 'norm_matrix'
  :rtype: float
  =#

  if norm_matrix === nothing
      norm_matrix = float(I(size(vec1)[1]))
  end

  return sum(sum(vec1' * norm_matrix * vec2))

end


function mynorm(vec::Array, norm_matrix = nothing)
  #= It computes the norm of 'vec', defined by the (positive definite) matrix 'norm_matrix'.
  If 'norm_matrix' is None (default), the Euclidean norm of 'vec' is returned.
  :param vec: vector
  :type vec: np.ndarray
  :param norm_matrix: positive definite matrix, defining the norm. If None, it defaults to the identity.
  :type norm_matrix: scipy.sparse.csc_matrix or np.ndarray or NoneType
  :return: norm of 'vec', defined by 'norm_matrix'
  :rtype: float
  =#

  if norm_matrix === nothing
    norm_matrix = float(I(size(vec)[1]))
  end

  return sqrt(mydot(vec, vec, norm_matrix))

end

function generate_parameter(a::T, b::T, n::Int64 = 1) where T <: Array{Float64}

  return [[rand(Uniform(a[i], b[i])) for i = 1:length(a)] for j in 1:n]

end

function generate_vtk_file_transient(Ω::BodyFittedTriangulation, path::String, var_name::String, var::SingleFieldFEFunction)

  createpvd("poisson_transient_solution") do pvd
    for (uₕ,t) in uₕₜ
      pvd[t] = createvtk(Ω, path * "var_name_$t" * ".vtu", cellfields = [var_name => var])
    end
  end

end
