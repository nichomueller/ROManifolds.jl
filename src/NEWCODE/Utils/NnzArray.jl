mutable struct NnzArray{T}
  nonzero_val::AbstractArray
  nonzero_idx::Vector{Int}
  nrows::Int
end

function compress(entire_array::T) where {T<:AbstractMatrix}
  nonzero_idx,nonzero_val = compress_array(entire_array)
  nrows = size(entire_array,1)
  NnzArray{T}(nonzero_val,nonzero_idx,nrows)
end

function compress(entire_arrays::Vector{T}) where {T<:AbstractArray}
  entire_array = hcat(entire_arrays...)
  compress(entire_array)
end

function Base.hcat(nza_vec::Vector{NnzArray{T}}) where T
  msg = """\n
  Cannot hcat the given NnzArrays: the nonzero indices and/or the full
  order number of rows do not match one another.
  """

  test_nnz_idx = nza_vec[1].nonzero_idx
  test_nrows = nza_vec[1].nrows
  @assert all([nza.nonzero_idx == test_nnz_idx for nza in nza_vec]) msg
  @assert all([nza.nrows == test_nrows for nza in nza_vec]) msg

  nza = hcat([nza.nonzero_val for nza in nza_vec]...)

  NnzArray{T}(nza,test_nnz_idx,test_nrows)
end

Base.size(nza::NnzArray,idx...) = size(nza.nonzero_val,idx...)

Base.getindex(nza::NnzArray,idx...) = nza.nonzero_val[idx...]

Base.eachindex(nza::NnzArray) = eachindex(nza.nonzero_val)

Base.setindex!(nza::NnzArray,val,idx...) = setindex!(nza.nonzero_val,val,idx...)

function Base.show(io::IO,nza::NnzArray)
  print(io,"NnzArray storing $(length(nza.nonzero_idx)) nonzero values")
end

function Base.copy(nza::NnzArray{T}) where T
  NnzArray{T}(copy(nza.nonzero_val),copy(nza.nonzero_idx),copy(nza.nrows))
end

Base.copyto!(nza::NnzArray,val::AbstractArray) = copyto!(nza.nonzero_val,val)

function Base.:(*)(nza1::NnzArray{T},nza2::NnzArray{T}) where T
  msg = """\n
  Cannot multiply the given NnzArray, the nonzero indices and/or the full
  order number of rows do not match one another.
  """
  @assert nza1.nonzero_idx == nza2.nonzero_idx msg
  @assert nza1.nrows == nza2.nrows msg
  mat = nza1.nonzero_val*nza2.nonzero_val
  NnzArray{T}(mat,copy(nza1.nonzero_idx),copy(nza1.nrows))
end

function Gridap.FESpaces.allocate_matrix(nza::NnzArray,sizes...)
  allocate_matrix(nza.nonzero_val,sizes...)
end

function adjoint!(nza::NnzArray)
  nza.nonzero_val = nza.nonzero_val'
  return
end

function convert!(::Type{T},nza::NnzArray) where T
  nza.nonzero_val = convert(T,nza.nonzero_val)
  return
end

function reshape!(nza::NnzArray{T},size...) where T
  nza.nonzero_val = reshape(nza.nonzero_val,size...)
  return
end

function recast(nza::NnzArray{<:AbstractMatrix})
  entire_array = zeros(nza.nrows,size(nza,2))
  entire_array[nza.nonzero_idx,:] = nza.nonzero_val
  entire_array
end

function recast(nza::NnzArray{<:SparseMatrixCSC},col=1)
  sparse_rows,sparse_cols = from_vec_to_mat_idx(nza.nonzero_idx,nza.nrows)
  ncols = maximum(sparse_cols)
  sparse(sparse_rows,sparse_cols,nza.nonzero_val[:,col],nza.nrows,ncols)
end

function recast(nza::NnzArray{<:SparseMatrixCSC})
  nvec = size(nza.nonzero_val,2)
  entire_array = Vector{SparseMatrixCSC{Float64,Int}}(undef,nvec)
  for col in axes(nza.nonzero_val,2)
    setindex!(entire_array,recast(nza,col),col)
  end
  entire_array
end

function change_mode!(nza::NnzArray,nparams::Int)
  mode1_ndofs = size(nza,1)
  mode2_ndofs = Int(size(nza,2)/nparams)

  mode2 = reshape(similar(nza.nonzero_val),mode2_ndofs,mode1_ndofs*nparams)
  _mode2(k::Int) = nza.nonzero_val[:,(k-1)*mode2_ndofs+1:k*mode2_ndofs]'
  @inbounds for k = 1:nparams
    setindex!(mode2,_mode2(k),:,(k-1)*mode1_ndofs+1:k*mode1_ndofs)
  end

  nza.nonzero_val = mode2
  return
end

function change_mode(nza::NnzArray,nparams::Int)
  nzm_copy = copy(nza)
  change_mode!(nzm_copy,nparams)
  nzm_copy
end

function tpod!(nza::NnzArray;kwargs...)
  nza.nonzero_val = tpod(nza.nonzero_val;kwargs...)
  return
end

function tpod(nza::NnzArray;kwargs...)
  nzm_copy = copy(nza)
  tpod!(nzm_copy;kwargs...)
  nzm_copy
end
