function Gridap.Algebra.allocate_matrix(::Type{T}) where T
  Matrix{T}(undef,0,0)
end

function Gridap.Algebra.allocate_vector(::Type{T}) where T
  Vector{T}(undef,0)
end

function allocate_mblock(::Type{T}) where T
  Matrix{T}[]
end

function allocate_vblock(::Type{T}) where T
  Vector{T}[]
end

function fill_rows_with_zeros(M::Matrix,row_idx::Vector{Int})
  r = size(M)[1]+length(idx)
  c = size(M)[2]
  col_idx = collect(1:c)

  sparse(row_idx,col_idx,M,r,c)
end
