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

function shift(A::AbstractMatrix,indices::AbstractVector,ns::Integer)
  A_shift = zeros(eltype(A),size(A))
  for i in indices
    A_shift[(i-1)*ns+1:i*ns,:] = A[(i-1)*ns+1:i*ns,:]
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
