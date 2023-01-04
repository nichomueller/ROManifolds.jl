"""Given a vector 'idx' referred to the entries of a vector 'vec' of length Ns^2,
 this function computes the row-column indexes of the NsxNs matrix associated to 'vec'"""
function from_vec_to_mat_idx(idx::Vector{Int},Ns::Int)
  col_idx = 1 .+ Int.(floor.((idx.-1)/Ns))
  findnz_map = idx - (col_idx.-1)*Ns
  findnz_map,col_idx
end

"""Given a sparse NsxC matrix Msparse and its NfullxC full representation Mfull,
 obtained by removing the zero-valued rows of Msparse, this function takes as
 input the vector of indexes 'full_idx' (referred to Mfull) and returns the vector
 of indexes 'sparse_idx' (referred to Msparse)"""
function from_full_idx_to_sparse_idx(
  full_idx::Vector{Int},
  sparse_to_full_idx::Vector{Int},
  Ns::Int)

  Nfull = length(sparse_to_full_idx)
  full_idx_space,full_idx_time = from_vec_to_mat_idx(full_idx, Nfull)
  (full_idx_time.-1)*Ns^2+sparse_to_full_idx[full_idx_space]
end

function from_sparse_idx_to_full_idx(
  sparse_idx::Int,
  sparse_to_full_idx::Vector{Int})

  findall(x -> x == sparse_idx, sparse_to_full_idx)[1]
end

function from_sparse_idx_to_full_idx(
  sparse_idx::Vector{Int},
  sparse_to_full_idx::Vector{Int})

  sparse_to_full(sidx) = from_sparse_idx_to_full_idx(sidx,sparse_to_full_idx)
  sparse_to_full.(sparse_idx)
end

function spacetime_idx(
  space_idx::Vector{Int},
  time_idx::Vector{Int},
  Ns=maximum(space_idx))

  (time_idx .- 1)*Ns .+ space_idx
end

function Base.argmax(v::Vector,nval::Int)
  s = sort(v,rev=true)
  Int.(indexin(s,v))[1:nval]
end
