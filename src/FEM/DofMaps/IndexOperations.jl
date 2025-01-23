"""
    recast_indices(indices::AbstractVector,a::AbstractSparseMatrix) -> AbstractVector

Given a sparse matrix `a` of size (M,N) and a number of nonzero entries Nnz,
recasts an array of indices ∈ {1,...,Nnz} to an array of entire_indices ∈ {1,...,M⋅N}
"""
function recast_indices(indices::AbstractVector,a::AbstractSparseMatrix)
  indices′ = copy(indices)
  I,J, = findnz(a)
  nrows = size(a,1)
  for (i,indi) in enumerate(indices′)
    indices′[i] = I[indi] + (J[indi]-1)*nrows
  end
  return indices′
end

"""
    sparsify_indices(indices::AbstractVector,a::AbstractSparseMatrix) -> AbstractVector

Inverse map of recast_indices.
"""
function sparsify_indices(indices::AbstractVector,a::AbstractSparseMatrix)
  indices′ = copy(indices)
  I,J, = findnz(a)
  nrows = size(a,1)
  for (i,indi) in enumerate(indices′)
    for k in nnz(a)
      i′ = I[k] + (J[k]-1)*nrows
      if i′ == indi
        indices′[i] = k
        break
      end
    end
  end
  return indices′
end

"""
    to_nz_index(i::AbstractArray,a::AbstractSparseMatrix) -> AbstractArray

Given a sparse matrix `a` representing a M×N sparse matrix with Nnz nonzero
entries, and a D-array of indices `i` with values in {1,...,MN}, returns the
corresponding D-array of indices with values in {1,...,Nnz}. A -1 is placed
whenever an entry of `i` corresponds to a zero entry of the sparse matrix
"""
function to_nz_index(i::AbstractArray,a::AbstractSparseMatrix)
  i′ = copy(i)
  to_nz_index!(i′,a)
  return i′
end

function to_nz_index!(i::AbstractArray,a::AbstractSparseMatrix)
  @abstractmethod
end

function to_nz_index!(i::AbstractArray,a::SparseMatrixCSC)
  nrows = size(a,1)
  for (j,index) in enumerate(i)
    if index > 0
      irow = fast_index(index,nrows)
      icol = slow_index(index,nrows)
      i[j] = nz_index(a,irow,icol)
    end
  end
end

# to_nz_index in case we don't provide a sparse matrix

function to_nz_index(i::AbstractArray)
  i′ = similar(i)
  nz_sortperm!(i′,i)
  i′
end

function nz_sortperm!(i′::AbstractArray,i::AbstractArray)
  fill!(i′,zero(eltype(i′)))
  inz = findall(!iszero,i)
  anz = sortperm(i[inz])
  inds = LinearIndices(size(anz))
  i′[inz[anz]] = inds
end

"""
    slow_index(i,nfast::Integer) -> Any

Returns the slow index in a tensor product structure. Suppose we have two matrices
A and B of sizes Ra × Ca and Rb × Rb. Their kronecker product AB = A ⊗ B, of size
RaRb × CaCb, can be indexed as AB[i,j] = A[slow_index(i,RbCb)] * B[fast_index(i,RbCb)]
"""
@inline slow_index(i,nfast::Integer) = cld.(i,nfast)
@inline slow_index(i::Integer,nfast::Integer) = cld(i,nfast)
@inline slow_index(i::Colon,::Integer) = i

"""
    fast_index(i,nfast::Integer) -> Any

Returns the fast index in a tensor product structure. Suppose we have two matrices
A and B of sizes Ra × Ca and Rb × Rb. Their kronecker product AB = A ⊗ B, of size
RaRb × CaCb, can be indexed as AB[i,j] = A[slow_index(i,RbCb)] * B[fast_index(i,RbCb)]

"""
@inline fast_index(i,nfast::Integer) = mod.(i .- 1,nfast) .+ 1
@inline fast_index(i::Integer,nfast::Integer) = mod(i - 1,nfast) + 1
@inline fast_index(i::Colon,::Integer) = i

recast(a::AbstractArray,i::AbstractArray) = @abstractmethod
sparsify(a::AbstractArray,i::AbstractArray) = @abstractmethod
