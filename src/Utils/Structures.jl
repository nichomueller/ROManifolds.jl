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

function SparseArrays.sparsevec(M::Matrix{T},row_idx::Vector{Int}) where T
  sparse_vblocks = SparseVector{T}[]
  for j = axes(M,2)
    push!(sparse_vblocks,sparsevec(row_idx,M[:,j],maximum(row_idx)))
  end

  sparse_vblocks
end

function sparsevec_to_sparsemat(svec::SparseVector,Nc::Int)
  ij,v = findnz(svec)
  i,j = from_vec_to_mat_idx(ij,Nc)
  sparse(i,j,v,maximum(i),Nc)
end
