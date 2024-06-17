function recast_indices(indices::AbstractVector,A::AbstractArray)
  nonzero_indices = get_nonzero_indices(A)
  entire_indices = nonzero_indices[indices]
  return entire_indices
end

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

@inline slow_index(i,N::Integer) = cld.(i,N)
@inline slow_index(i::Colon,::Integer) = i
@inline fast_index(i,N::Integer) = mod.(i .- 1,N) .+ 1
@inline fast_index(i::Colon,::Integer) = i

function tensorize_indices(i::Integer,dofs::AbstractVector{<:Integer})
  D = length(dofs)
  cdofs = cumprod(dofs)
  ic = ()
  @inbounds for d = 1:D-1
    ic = (ic...,fast_index(i,cdofs[d]))
  end
  ic = (ic...,slow_index(i,cdofs[D-1]))
  return CartesianIndex(ic)
end

function tensorize_indices(indices::AbstractVector,dofs::AbstractVector{<:Integer})
  D = length(dofs)
  tindices = Vector{CartesianIndex{D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    tindices[ii] = tensorize_indices(i,dofs)
  end
  return tindices
end

function split_row_col_indices(i::CartesianIndex{D},dofs::AbstractMatrix{<:Integer}) where D
  @check size(dofs) == (2,D)
  nrows = view(dofs,:,1)
  irc = ()
  @inbounds for d = 1:D
    irc = (irc...,fast_index(i.I[d],nrows[d]),slow_index(i.I[d],nrows[d]))
  end
  return CartesianIndex(irc)
end

function split_row_col_indices(indices::AbstractVector{CartesianIndex{D}},dofs::AbstractMatrix{<:Integer}) where D
  rcindices = Vector{CartesianIndex{2*D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    rcindices[ii] = split_row_col_indices(i,dofs)
  end
  return rcindices
end
