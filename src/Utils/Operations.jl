function expand(tup::Tuple)
  t = ()
  for el = tup
    if isa(el,Tuple)
      t = (t...,expand(el)...)
    else
      t = (t...,el)
    end
  end
  t
end

# function SparseArrays.findnz(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
#   numnz = nnz(S)
#   I = Vector{Ti}(undef,numnz)
#   J = Vector{Ti}(undef,numnz)
#   V = Vector{Tv}(undef,numnz)
#   count = 1
#   @inbounds for col = 1:size(S,2), k = SparseArrays.getcolptr(S)[col]:(SparseArrays.getcolptr(S)[col+1]-1)
#     I[count] = rowvals(S)[k]
#     J[count] = col
#     V[count] = nonzeros(S)[k]
#     count += 1
#   end
#   nz = findall(x -> x .>= eps(),abs.(V))
#   (I[nz],J[nz],V[nz])
# end

# function SparseArrays.findnz(x::SparseVector{Tv,Ti}) where {Tv,Ti}
#   numnz = nnz(x)
#   I = Vector{Ti}(undef, numnz)
#   V = Vector{Tv}(undef, numnz)
#   nzind = SparseArrays.nonzeroinds(x)
#   nzval = nonzeros(x)
#   @inbounds for i = 1:numnz
#     I[i] = nzind[i]
#     V[i] = nzval[i]
#   end
#   nz = findall(v -> abs.(v) .>= eps(), V)
#   (I[nz], V[nz])
# end

function compress_array(entire_array::AbstractVector)
  nonzero_idx = findall(x -> abs(x) ≥ eps(),entire_array)
  nonzero_idx,entire_array[nonzero_idx]
end

function compress_array(entire_array::AbstractMatrix)
  sum_cols = vec(sum(entire_array,dims=2))
  nonzero_idx = findall(x -> abs(x) ≥ eps(),sum_cols)
  nonzero_idx,entire_array[nonzero_idx,:]
end

ℓ∞(x) = maximum(abs.(x))
