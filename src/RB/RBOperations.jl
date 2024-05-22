function recast_indices(A::AbstractArray,indices::AbstractVector)
  nonzero_indices = get_nonzero_indices(A)
  entire_indices = nonzero_indices[indices]
  return entire_indices
end

function recast_indices(b::TTSVDCores,indices::AbstractVector)
  space_dofs = _num_tot_space_dofs(b)
  tensor_indices = tensorize_indices(space_dofs,indices)
  return tensor_indices
end

function recast_indices(b::MatrixTTSVDCores,indices::AbstractVector)
  space_dofs = _num_tot_space_dofs(b)
  tensor_indices = tensorize_indices(vec(prod(space_dofs;dims=1)),indices)
  return _split_row_col(space_dofs,tensor_indices)
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

function tensorize_indices(dofs::Vector{Int},indices::AbstractVector)
  D = length(dofs)
  cdofs = cumprod(dofs)
  tindices = Vector{CartesianIndex{D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    ic = ()
    @inbounds for d = 1:D-1
      ic = (ic...,fast_index(i,cdofs[d]))
    end
    ic = (ic...,slow_index(i,cdofs[D-1]))
    tindices[ii] = CartesianIndex(ic)
  end
  return tindices
end

function _split_row_col(dofs::Matrix{Int},tindices::AbstractVector{CartesianIndex{D}}) where D
  @check size(dofs) == (2,D)
  nrows = view(dofs,1,:)
  tids_row_col = Vector{CartesianIndex{2*D}}(undef,length(tindices))
  @inbounds for (ii,i) = enumerate(tindices)
    irc = ()
    @inbounds for d = 1:D
      irc = (irc...,fast_index(i.I[d],nrows[d]),slow_index(i.I[d],nrows[d]))
    end
    tids_row_col[ii] = CartesianIndex(irc)
  end
  return tids_row_col
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
