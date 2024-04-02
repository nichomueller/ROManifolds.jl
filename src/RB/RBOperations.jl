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

function get_nonzero_indices(A::TTMatrix)
  return get_nonzero_indices(A.values)
end

function get_nonzero_indices(A::AbstractArray{T,3} where T)
  return axes(A,2)
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
