function recast_indices(A::AbstractMatrix,indices::AbstractVector)
  return indices
end

function sparsify_indices(A::AbstractMatrix,indices::AbstractVector)
  return indices
end

function recast_indices(A::AbstractSparseMatrix,indices::AbstractVector)
  nonzero_indices = get_nonzero_indices(A)
  entire_indices = nonzero_indices[indices]
  return entire_indices
end

function sparsify_indices(A::AbstractSparseMatrix,indices::AbstractVector)
  nonzero_indices = get_nonzero_indices(A)
  sparse_indices = map(y->findfirst(x->x==y,nonzero_indices),indices)
  return sparse_indices
end

function recast_indices(A::TTSparseMatrix,indices::AbstractVector)
  recast_indices(A.values,indices)
end

function sparsify_indices(A::TTSparseMatrix,indices::AbstractVector)
  sparsify_indices(A.values,indices)
end

function get_nonzero_indices(A::AbstractSparseMatrix)
  i,j, = findnz(A)
  return i .+ (j .- 1)*A.m
end

function get_nonzero_indices(A::TTSparseMatrix)
  get_nonzero_indices(A.values)
end

function compress_basis_space(A::AbstractMatrix,B::AbstractMatrix)
  map(eachcol(A)) do a
    B'*a
  end
end

function compress_basis_space(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix)
  map(get_values(A)) do A
    C'*A*B
  end
end

function combine_basis_time(A::AbstractMatrix;kwargs...)
  A
end

function combine_basis_time(A::AbstractMatrix,B::AbstractMatrix;combine=(x,y)->x)
  time_ndofs = size(B,1)
  nt_row = size(B,2)
  nt_col = size(A,2)

  T = eltype(A)
  bt_proj = zeros(T,time_ndofs,nt_row,nt_col)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    bt_proj[:,it,jt] .= B[:,it].*A[:,jt]
    bt_proj_shift[2:end,it,jt] .= B[2:end,it].*A[1:end-1,jt]
  end

  combine(bt_proj,bt_proj_shift)
end

function _shift(A::AbstractMatrix,ns::Integer,style=:forwards)
  A_shift = zeros(eltype(A),size(A))
  for i in 2:floor(Int,size(A,1)/ns)
    ids_forwards = (i-1)*ns+1:i*ns
    ids_backwards = (i-2)*ns+1:(i-1)*ns
    Ai = style == :forwards ? A[ids_forwards,:] : A[ids_backwards,:]
    A_shift[ids_forwards,:] = Ai
  end
  return A_shift
end

function compress_combine_basis_space_time(A,B;kwargs...)
  map(eachcol(A)) do a
    B'*a
  end
end

function compress_combine_basis_space_time(A,B,C,B_shift,C_shift;combine=(x,y)->x)
  map(get_values(A)) do A
    combine(C'*A*B,C_shift'*A*B_shift)
  end
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
