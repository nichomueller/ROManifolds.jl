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

  T = eltype(get_vector_type(test))
  bt_proj = zeros(T,time_ndofs,nt_row,nt_col)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    bt_proj[:,it,jt] .= B[:,it].*A[:,jt]
    bt_proj_shift[2:end,it,jt] .= B[2:end,it].*A[1:end-1,jt]
  end

  combine(bt_proj,bt_proj_shift)
end
