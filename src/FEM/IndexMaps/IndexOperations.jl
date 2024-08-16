"""
    recast_indices(indices::AbstractVector,A::AbstractArray) -> AbstractVector

Recasts an array of indices ∈ {1,...,nnz(A)} to an array of entire_indices ∈ {1,...,length(A)}
when A is a sparse matrix, and it returns the original vector itself when A is
an ordinary array.

"""
function recast_indices(indices::AbstractVector,A::AbstractArray)
  indices′ = copy(indices)
  nonzero_indices = get_nonzero_indices(A)
  for (i,indi) in enumerate(indices′)
    indices′[i] = nonzero_indices[indi]
  end
  return indices′
end

"""
    sparsify_indices(indices::AbstractVector,A::AbstractArray) -> AbstractVector

Inverse map of recast_indices.

"""
function sparsify_indices(indices::AbstractVector,A::AbstractArray)
  indices′ = copy(indices)
  nonzero_indices = get_nonzero_indices(A)
  for (i,indi) in enumerate(indices′)
    indices′[i] = findfirst(x->x==indi,nonzero_indices)
  end
  return indices′
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
