mutable struct NnzArray{T}
  array::AbstractArray
  nonzero_idx::Vector{Int}
  nrows::Int
end

function NnzArray(entire_array::T) where {T<:AbstractMatrix}
  nonzero_idx,array = compress(entire_array)
  nrows = size(entire_array,1)
  NnzArray{T}(array,nonzero_idx,nrows)
end

function NnzArray(arrays::Vector{T}) where {T<:AbstractArray}
  entire_array = hcat(arrays...)
  NnzArray(entire_array)
end

Base.size(nza::NnzArray,idx...) = size(nza.array,idx...)

Base.getindex(nza::NnzArray,idx...) = nza.array[idx...]

Base.eachindex(nza::NnzArray) = eachindex(nza.array)

Base.setindex!(nza::NnzArray,val,idx...) = setindex!(nza.array,val,idx...)

function Base.copy(nza::NnzArray{T}) where T
  NnzArray{T}(copy(nza.array),copy(nza.nonzero_idx),copy(nza.nrows))
end

Base.copyto!(nza::NnzArray,val::AbstractArray) = copyto!(nza.array,val)

function Base.show(io::IO,nmz::NnzArray{T}) where T
  l = length(nmz.nonzero_idx)
  print(io,"NnzArray{$T} computed from a matrix with $l nonzero row entries")
end

function Base.:(*)(nza1::NnzArray{T},nza2::NnzArray{T}) where T
  msg = """\n
  Cannot multiply the given Nnzaatrices, the nonzero indices and/or the full
  order number of rows do not match one another.
  """
  @assert nza1.nonzero_idx == nza2.nonzero_idx msg
  @assert nza1.nrows == nza2.nrows msg
  array = nza1.array*nza2.array
  NnzArray{T}(array,copy(nza1.nonzero_idx),copy(nza1.nrows))
end

function Base.adjoint(nza::NnzArray{T}) where T
  array = nza.array'
  NnzArray{T}(array,copy(nza.nonzero_idx),copy(nza.nrows))
end

function Gridap.FESpaces.allocate_matrix(nza::NnzArray,sizes...)
  allocate_matrix(nza.array,sizes...)
end

function convert!(::Type{T},nza::NnzArray) where T
  nza.array = convert(T,nza.array)
  nza
end

function compress(entire_array::AbstractMatrix)
  sum_cols = reshape(sum(entire_array,dims=2),:)
  nonzero_idx = findall(x -> abs(x) ≥ eps(),sum_cols)
  nonzero_idx,entire_array[nonzero_idx,:]
end

function compress(entire_array::SparseMatrixCSC{Float,Int})
  findnz(entire_array[:])
end

function compress(nza::Vector{NnzArray{T}};type=EMatrix{Float}) where T
  msg = """\n
  Cannot compress the given NnzArrays, the nonzero indices and/or the full
  order number of rows do not match one another.
  """

  test_nnz_idx,test_nrows = nza[1].nonzero_idx,nza[1].nrows
  @assert all([m.nonzero_idx == test_nnz_idx for m in nza]) msg
  @assert all([m.nrows == test_nrows for m in nza]) msg

  array = hcat([m.array for m in nza]...)
  conv_array = convert(type,array)
  NnzArray{T}(conv_array,test_nnz_idx,test_nrows)
end

function compress(nza::Vector{Vector{NnzArray{T}}};kwargs...) where T
  sorted_nza(i) = map(m->getindex(m,i),nza)
  map(i -> compress(sorted_nza(i);kwargs...),eachindex(nza))
end

function recast(nza::NnzArray{<:AbstractMatrix})
  entire_array = zeros(nza.nrows,size(nza,2))
  entire_array[nza.nonzero_idx,:] = nza.array
  entire_array
end

function recast(nza::NnzArray{<:SparseMatrixCSC},col=1)
  sparse_rows,sparse_cols = from_vec_to_mat_idx(nza.nonzero_idx,nza.nrows)
  ncols = maximum(sparse_cols)
  sparse(sparse_rows,sparse_cols,nza.array[:,col],nza.nrows,ncols)
end

function change_mode!(nza::NnzArray,nparams::Int)
  mode1_ndofs = size(nza,1)
  mode2_ndofs = Int(size(nza,2)/nparams)

  mode2 = reshape(similar(nza.array),mode2_ndofs,mode1_ndofs*nparams)
  _mode2(k::Int) = nza.array[:,(k-1)*mode2_ndofs+1:k*mode2_ndofs]'
  @inbounds for k = 1:nparams
    setindex!(mode2,_mode2(k),:,(k-1)*mode1_ndofs+1:k*mode1_ndofs)
  end

  nza.array = mode2
  return
end

_compress_rows(nza::NnzArray) = size(nza.array,1) > size(nza.array,2)

function tpod!(nza::NnzArray;kwargs...)
  compress_rows = _compress_rows(nza)
  nza.array = tpod(Val{compress_rows}(),nza.array;kwargs...)
end

function tpod(::Val{true},array::AbstractMatrix;ϵ=1e-4)
  compressed_array = array'*array
  _,Σ2,V = svd(compressed_array)
  Σ = sqrt.(Σ2)
  n = truncation(Σ,ϵ)
  U = array*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (Σ[i]+eps())
  end
  U
end

function tpod(::Val{false},array::AbstractMatrix;ϵ=1e-4)
  compressed_array = array*array'
  U,Σ2,_ = svd(compressed_array)
  Σ = sqrt.(Σ2)
  n = truncation(Σ,ϵ)
  U[:,1:n]
end

function truncation(Σ::AbstractArray,ϵ::Real)
  energies = cumsum(Σ.^2;dims=1)
  rb_ndofs = first(findall(x->x ≥ (1-ϵ^2)*energies[end],energies))[1]
  err = sqrt(1-energies[rb_ndofs]/energies[end])
  printstyled("POD truncated at ϵ = $ϵ: number basis vectors = $rb_ndofs; projection error ≤ $err\n";
    color=:blue)
  rb_ndofs
end
