"""
    recast_indices(indices::AbstractVector,A::AbstractArray) -> AbstractVector

Recasts an array of indices ∈ {1,...,nnz(A)} to an array of entire_indices ∈ {1,...,length(A)}
when A is a sparse matrix, and it returns the original vector itself when A is
an ordinary array.
"""

function recast_indices(indices::AbstractVector,A::AbstractArray)
  nonzero_indices = get_nonzero_indices(A)
  entire_indices = nonzero_indices[indices]
  return entire_indices
end

"""
    sparsify_indices(indices::AbstractVector,A::AbstractArray) -> AbstractVector

Inverse map of recast_indices.
"""

function sparsify_indices(indices::AbstractVector,A::AbstractArray)
  nonzero_indices = get_nonzero_indices(A)
  sparse_indices = map(y->findfirst(x->x==y,nonzero_indices),indices)
  return sparse_indices
end

function get_nonzero_indices(A::AbstractArray)
  @notimplemented
end

function get_nonzero_indices(A::AbstractMatrix)
  return axes(A,1)
end

function get_nonzero_indices(A::AbstractSparseMatrix)
  i,j, = findnz(A)
  return i .+ (j .- 1)*A.m
end

function get_nonzero_indices(A::AbstractArray{T,3} where T)
  return axes(A,2)
end

"""
    slow_index(i,nfast::Integer) -> Any

Returns the slow index in a tensor product structure. Suppose we have two matrices
A and B of sizes Ra × Ca and Rb × Rb. Their kronecker product AB = A ⊗ B, of size
RaRb × CaCb, can be indexed as AB[i,j] = A[slow_index(i,RbCb)] * B[fast_index(i,RbCb)]
"""

@inline slow_index(i,nfast::Integer) = cld.(i,nfast)
@inline slow_index(i::Colon,::Integer) = i

"""
    fast_index(i,nfast::Integer) -> Any

Returns the fast index in a tensor product structure. Suppose we have two matrices
A and B of sizes Ra × Ca and Rb × Rb. Their kronecker product AB = A ⊗ B, of size
RaRb × CaCb, can be indexed as AB[i,j] = A[slow_index(i,RbCb)] * B[fast_index(i,RbCb)]
"""

@inline fast_index(i,nfast::Integer) = mod.(i .- 1,nfast) .+ 1
@inline fast_index(i::Colon,::Integer) = i

"""
    tensorize_indices(i::Integer,s::NTuple{D,Integer}) where D -> CartesianIndex{D}

Given the size s of a D-dimensional array, converts the index i from a IndexLinear
style to a IndexCartesian style.
"""

function tensorize_indices(i::Integer,s::NTuple{D,Integer}) where D
  D = length(s)
  cdofs = cumprod(s)
  ic = ()
  @inbounds for d = 1:D-1
    ic = (ic...,fast_index(i,cdofs[d]))
  end
  ic = (ic...,slow_index(i,cdofs[D-1]))
  return CartesianIndex(ic)
end

function tensorize_indices(indices::AbstractVector,s::NTuple{D,Integer}) where D
  tindices = Vector{CartesianIndex{D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    tindices[ii] = tensorize_indices(i,s)
  end
  return tindices
end
