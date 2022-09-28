"""Given a vector 'idx' referred to the entries of a vector 'vec' of length Nₕ^2,
 this function computes the row-column indexes of the NₕxNₕ matrix associated to 'vec'"""
function from_vec_to_mat_idx(idx::Vector{Int}, Nₕ::Int)
  col_idx = 1 .+ floor.(Int,(idx.-1)/Nₕ)
  row_idx = idx - (col_idx.-1)*Nₕ
  row_idx, col_idx
end

function from_vec_to_mat_idx(idx::Vector{Vector{Int}}, Nₕ::Int)

  row_idx, col_idx = Vector{Int}[], Vector{Int}[]
  for nb = eachindex(idx)
    push!(row_idx, from_vec_to_mat_idx(idx[nb], Nₕ)[1])
    push!(col_idx, from_vec_to_mat_idx(idx[nb], Nₕ)[2])
  end

  row_idx, col_idx

end

"""Given a sparse NₕxC matrix Msparse and its NfullxC full representation Mfull,
 obtained by removing the zero-valued rows of Msparse, this function takes as
 input the vector of indexes 'full_idx' (referred to Mfull) and returns the vector
 of indexes 'sparse_idx' (referred to Msparse)"""
function from_full_idx_to_sparse_idx(
  full_idx::Vector{Int},
  sparse_to_full_idx::Vector{Int},
  Nₕ::Int)

  Nfull = length(sparse_to_full_idx)
  full_idx_space,full_idx_time = from_vec_to_mat_idx(full_idx, Nfull)
  (full_idx_time.-1)*Nₕ^2+sparse_to_full_idx[full_idx_space]

end

function from_full_idx_to_sparse_idx(
  full_idx::Vector{Vector{Int}},
  sparse_to_full_idx::Vector{Int},
  Nₕ::Int)

  sparse_idx = Vector{Int}[]
  for nb = eachindex(full_idx)
    push!(sparse_idx,
      from_full_idx_to_sparse_idx(full_idx[nb], sparse_to_full_idx, Nₕ))
  end

  sparse_idx

end

"""Removes zero-valued rows of a CSC matrix Msparse and returns its full
  representation Mfull"""
function remove_zero_entries(Msparse::SparseMatrixCSC{T}) where T
  for col = 1:size(Msparse)[2]
    _,vals = findnz(Msparse[:,col])::Tuple{Vector{Int},Vector{T}}
    if col == 1
      global Mfull = zeros(length(vals),size(Msparse)[2])
    end
    Mfull[:,col] = vals
  end
  T.(Mfull)
end

"""Given an unsorted vector 'vec', returns the vector of labels
{1,...,length(unique(vec))} that order the entries of 'vec'."""
function label_sorted_elems(vec::Vector{T}) where T
  vecnew = unique(sort(copy(vec)))
  Int.(indexin(vec,vecnew))
end

function Base.argmax(v::Vector{T},n_val::Int) where T
  s = sort(v,rev=true)
  idx = Int.(indexin(s,v))[1:n_val]
end

function unique_block(vv::Vector{Vector{T}}) where T
  uvv = Vector{T}[]
  for b = eachindex(vv)
    push!(uvv, unique(vv[b]))
  end
  uvv
end
