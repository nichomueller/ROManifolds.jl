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

function SparseArrays.findnz(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
  numnz = nnz(S)
  I = Vector{Ti}(undef,numnz)
  J = Vector{Ti}(undef,numnz)
  V = Vector{Tv}(undef,numnz)

  count = 1
  @inbounds for col = 1:size(S,2), k = SparseArrays.getcolptr(S)[col] : (SparseArrays.getcolptr(S)[col+1]-1)
      I[count] = rowvals(S)[k]
      J[count] = col
      V[count] = nonzeros(S)[k]
      count += 1
  end

  nz = findall(x -> x .>= eps(),V)

  (I[nz],J[nz],V[nz])
end

function compress_array(entire_array::AbstractVector)
  nonzero_idx = findall(x -> abs(x) ≥ eps(),entire_array)
  nonzero_idx,entire_array[nonzero_idx]
end

function compress_array(entire_array::AbstractMatrix)
  sum_cols = reshape(sum(entire_array,dims=2),:)
  nonzero_idx = findall(x -> abs(x) ≥ eps(),sum_cols)
  nonzero_idx,entire_array[nonzero_idx,:]
end

function compress_array(entire_array::SparseMatrixCSC{Float,Int})
  findnz(entire_array[:])
end

struct LazyArrayWrap{L}
  a::LazyArray
  function LazyArrayWrap(a::LazyArray)
    L = length(first(a.args))
    new{L}(a)
  end
end

Base.length(::LazyArrayWrap{L}) where L = L
Base.getindex(w::LazyArrayWrap,idx...) = getindex(w.a,idx...)

@generated function Base.hcat(w::LazyArrayWrap{D}) where D
  str = join(["w[$i], " for i in 1:D])
  Meta.parse("hcat($str)")
end

function Base.collect(a::LazyArray)
  w_a = LazyArrayWrap(a)
  hcat(w_a)
end
