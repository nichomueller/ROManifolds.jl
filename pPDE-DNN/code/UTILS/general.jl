using Pkg
Pkg.activate(".")

function create_dir(path)
    #=Create a directory at the given path
    :param name: path of the directory to be created, if not already existing
    :type name: str
    =#

    if !isdir(path)
        mkdir(path)
    end
    
end

function get_full_subdirectories(rootdir)
    #=Get a full list of subdirectories at a given root directory
    :param rootdir: path of the root directory whose subdirectories are searched and listed
    :type rootdir: str
    =#

    return filter(isdir, readdir(rootdir,join = true))

end


function save_variable(var, var_name, extension = "csv", path = nothing)
    #=Utility method, which allows to save arrays/matrices in .jld files. If the path to the text file does not exist,
    it displays an error message. 
    :param var: variable to be saved
    :type var: AbstractMatrix, AbstactVector
    :param variable_name: name with which we save the variable
    :type variable_name: str
    :param file_name: path to the file where the variable has to be saved, provided that the path is a valid path; no extension should be provided
    :type file_name: str
    =#

    if path === nothing
        path = pwd() * "/" * var_name * "." * extension
    end
    
    try
        if extension == "jld"
            save(path, var_name, var)
        else extension == "jld"            
            if issparse(var)
                i, j, v = findnz(var)
                df = DataFrame([:i => i, :j => j, :v => v])
                CSV.write(path, df)
            else
                CSV.write(path, Tables.table(var))
            end 
        end
    catch e
        println("Error: $e. Impossible to save the desired variable")
    end

end


function load_variable(var_name, extension = "csv", path = nothing, sparse = false, delimiter = nothing)
    #=Utility method, which allows to load arrays/matrices from .jld files. If the path to the text file does not exist,
    it displays an error message.
    :param variable_name: name of the loaded variable
    :type variable_name: str
    :param file_name: path to the file where the variable has to be loaded from, provided that the path is a valid path; no extension should be provided
    :type file_name: str
    :return the variable read from the .jld file
    :rtype: AbstractMatrix, AbstactVector
    =#

    @assert !(sparse === false && (extension != "csv" || extension != "txt"))

    if path === nothing
        path = pwd() * "/" * var_name * "." * extension
    end

    try
        if extension == "jld"
            var = load(path, var_name)  

        elseif extension == "csv"
            var = Matrix(CSV.read(path, DataFrame))
            if sparse === true
                return sparse(convert(Vector{Int64}, var[:,1]), convert(Vector{Int64}, var[:,2]), var[:,3])
            end
        else
            if delimiter === nothing
                if sparse === true
                    py"""
                    i, j, v = np.loadtxt(path)
                    """
                    var = sparse(i, j, v)
                else
                    py"""
                    var = np.loadtxt(path)
                    """
                end
            else
                if sparse === true
                    py"""
                    i, j, v = np.genfromtxt(path, delimiter)
                    """
                    var = sparse(i, j, v)
                else
                    py"""
                    var = np.loadtxt(path, delimiter)
                    """
                end
            end
        end

        return var

    catch e
        println("Error: $e. Impossible to load the desired variable")
    end

    

end


function mydot(vec1, vec2, norm_matrix = nothing)
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
    return sum(sum(vec1' .* norm_matrix .* vec2)) 
end


function mynorm(vec, norm_matrix = nothing)
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
        norm_matrix = float(I(size(vec1)[1]))
    return sqrt(mydot(vec, vec, norm_matrix))
    end
end


function convert_to_sparse(var, format = "csc")
    #= It computes the sparse representation of the input variable 'var', either a matrix or a vector; 
    the format of the sparse variable is given by 'format'.
    :param mat: matrix, vector
    :type mat: AbstractMatrix, AbstractVector
    :param format: 'csc' (default), 'csr', 'coo'
    :type format: str
    :return: sparse representation of 'var', defined by 'sparse_var'
    :rtype: SparseOperatorCSC, SparseOperatorCSR, SparseOperatorCOO
    =#

    if format == "csc" || length(size(var)) == 1
        sparse_var = sparse(var)
    elseif format != "csc"
        if format == "csr" # works only when the number of nonzero elements is less than 10
            idx_cartesian = findall(!iszero, var)
            idx = hcat(getindex.(idx_cartesian, 1), getindex.(idx_cartesian,2))
            var_nnz = filter(el -> el != 0, var)
            sparse_var = sparsecsr(idx[:,1], idx[:,2], var_nnz)
        elseif format == "coo"
            idx_cartesian = findall(!iszero, var)
            idx = hcat(getindex.(idx_cartesian, 1), getindex.(idx_cartesian,2))
            var_nnz = filter(el -> el != 0, var)  
            sparse_var = CompressedSparseOperator{:COO}(var_nnz, idx[:,1], idx[:,2], max(idx[:,1]...), max(idx[:,2]...))
        end
    return sparse_var
    else
        @error "Error: unrecognized sparsity format - must choose between 'csc', 'csr', 'coo' "
        throw(ArgumentError)
    end
end


function sparse_matrix_matrix_mul(mat1, mat2, format = "csc")
    #=Computes the matrix multiplication between mat1 and mat2, assuming that mat1 is sparse and mat2 is full.
    mat1 can be either in 'raw' CO format or a scipy.sparse matrix.
    :param mat1: pre-multiplicative matrix, supposed to be sparse
    :type mat1: numpy.ndarray or scipy.sparse
    :param mat2: post-multiplicative matrix, supposed to be full
    :type mat2: numpy.ndarray
    :return: result of the sparse-full matrix multiplication mat1*mat2, given as a full matrix
    :rtype: numpy.ndarray
    =#

    if format == "csc" 
        M = mat1 * mat2
    elseif format in ("csr", "coo")
        info_mat1 = findnz(mat1)
        M = zeros(size(mat2))
        for i in range(1, length(info_mat1[1]))
            M[info_mat1[1][i], :] += info_mat1[3][i] * mat2[info_mat1[2][i], :]
        end
    else
        @error "Error: impossible to perform sparse matrix - full matrix multiplication"
        throw(TypeError)
    end
    return M
end


function sparse_matrix_vector_mul(mat, vec)
    #= Computes the matrix-vector multiplication between mat and vec, assuming that mat1 is sparse and vec is full.
    mat1 can be either in 'raw' CO format or a scipy.sparse matrix.
    :param mat: pre-multiplicative matrix, supposed to be sparse
    :type mat: numpy.ndarray or scipy.sparse
    :param vec: post-multiplicative vector, supposed to be full
    :type vec: numpy.ndarray
    :return: result of the sparse-full matrix-vector multiplication mat*vec, given as a full vector
    :rtype: numpy.ndarray
    =#

    if format == "csc" 
        V = mat * vec
    elseif format in ("csr", "coo")
        info_mat = findnz(mat)
        V = zeros(size(vec))
        for i in range(1, length(info_mat[1]))
            V[info_mat[1][i], :] += info_mat[3][i] * vec[info_mat[2][i]]
        end
    else
        @error "Error: impossible to perform sparse matrix - full vector multiplication"
        throw(TypeError)
    end
    return V
end


function sparse_to_full_matrix(mat, format = "csc")
    #=Method to convert a sparse matrix, given in COO format or as scipy.sparse matrix, to a full matrix,
    given its dimensions
    :param mat: input matrix, supposed to be sparse, to be converted to a full matrix
    :type mat: numpy.ndarray or scipy.sparse
    :param dims: dimensions of the final full matrix
    :type dims: tuple(int, int)
    :return: full version of the input matrix, of dimensions 'dims'
    :rtype: numpy.ndarray
    =#

    if format == "csc" 
        full_mat = Matrix(mat)
    elseif format in ("csr", "coo")
        info_mat= findnz(mat)
        full_mat = zeros(maximum(info_mat[1]), maximum(info_mat[2]))
        [full_mat[info_mat[1][i], info_mat[2][i]] = info_mat[3][i] for i = 1:length(info_mat[3])]
    else
        @error "Error: impossible to perform sparse matrix - full matrix multiplication"
        throw(TypeError)
    end
    return full_mat

end








#=
function main()
    #B = sparse(rand(5,5))
    #save_variable(B,"B","csv")
    #Bloaded = load_variable("B", "csv", nothing, true)

    mat1 = rand(10,10)
    aa = rand(10,10)
    mat2 = convert_to_sparse(aa, "coo")
    sparse_to_full_matrix(mat2, "coo")
    #sparse_matrix_matrix_mul(mat2, mat1, "coo")
    #sparse_matrix_matrix_mul(mat2, mat1[:,1], "coo")
end


main() =#