function recast_indices(A::AbstractArray,indices::AbstractVector)
  nonzero_indices = get_nonzero_indices(A)
  entire_indices = nonzero_indices[indices]
  return entire_indices
end

function sparsify_indices(A::AbstractArray,indices::AbstractVector)
  nonzero_indices = get_nonzero_indices(A)
  sparse_indices = map(y->findfirst(x->x==y,nonzero_indices),indices)
  return sparse_indices
end

function get_nonzero_indices(A::AbstractVector)
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

function tensorize_indices(i::Int,dofs::Vector{Int})
  D = length(dofs)
  cdofs = cumprod(dofs)
  ic = ()
  @inbounds for d = 1:D-1
    ic = (ic...,fast_index(i,cdofs[d]))
  end
  ic = (ic...,slow_index(i,cdofs[D-1]))
  return CartesianIndex(ic)
end

function tensorize_indices(indices::AbstractVector,dofs::AbstractVector{Int})
  D = length(dofs)
  tindices = Vector{CartesianIndex{D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    tindices[ii] = tensorize_indices(i,dofs)
  end
  return tindices
end

function split_row_col_indices(i::CartesianIndex{D},dofs::AbstractMatrix{Int}) where D
  @check size(dofs) == (2,D)
  nrows = view(dofs,:,1)
  irc = ()
  @inbounds for d = 1:D
    irc = (irc...,fast_index(i.I[d],nrows[d]),slow_index(i.I[d],nrows[d]))
  end
  return CartesianIndex(irc)
end

function split_row_col_indices(indices::AbstractVector{CartesianIndex{D}},dofs::AbstractMatrix{Int}) where D
  rcindices = Vector{CartesianIndex{2*D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    rcindices[ii] = split_row_col_indices(i,dofs)
  end
  return rcindices
end

function FEM.shift!(a::AbstractParamContainer,r::TransientParamRealization,α::Number,β::Number)
  b = copy(a)
  nt = num_times(r)
  np = num_params(r)
  @assert length(a) == nt*np
  for i = eachindex(a)
    it = slow_index(i,np)
    if it > 1
      a[i] .= α*a[i] + β*b[i-np]
    end
  end
end
