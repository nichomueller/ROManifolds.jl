mutable struct NnzMatrix{T}
  array::AbstractMatrix
  nonzero_idx::Vector{Int}
  nrows::Int

  function NnzMatrix(entire_array::T) where {T<:AbstractMatrix}
    array,nonzero_idx = compress(entire_array)
    nrows = size(entire_array,1)
    new{T}(array,nonzero_idx,nrows)
  end
end

Base.size(nzm::NnzMatrix) = size(nzm.array)

full_size(nzm::NnzMatrix) = (nzm.nrows,size(nzm,2))

Base.eltype(::Type{<:NnzMatrix{T}}) where T = T

Base.eltype(::NnzMatrix{T}) where T = T

Base.getindex(nzm::NnzMatrix,i...) = nzm.array[i...]

Base.eachindex(nzm::NnzMatrix) = eachindex(nzm.array)

Base.copyto!(nzm::NnzMatrix,val::AbstractMatrix) = copyto!(nzm.array,val)

function Base.copyto!(nzm::Vector{<:NnzMatrix},val::Vector{<:AbstractMatrix})
  @assert length(nzm) == length(val)
  map(copyto!,nzm,val)
end

Base.setindex!(nzm::NnzMatrix,val,idx) = setindex!(nzm.array,val,idx)

function Base.setindex!(nzm::Vector{<:NnzMatrix},val,idx)
  @assert length(nzm) == length(val)
  map((m,v) -> setindex!(m,v,idx),nzm,val)
end

function Base.show(io::IO,o::NnzMatrix)
  print(io,"NnzMatrix($(o.array), $(o.nonzero_idx))")
end

function convert!(::Type{T},nzm::NnzMatrix) where T
  nzm.array = convert(T,nzm.array)
  nzm
end

function compress(entire_array::AbstractMatrix)
  sum_cols = reshape(sum(entire_array,dims=2),:)
  nonzero_idx = findall(x -> abs(x) ≥ eps(),sum_cols)
  entire_array[nonzero_idx,:],nonzero_idx
end

function compress(entire_array::SparseMatrixCSC{Float,Int})
  findnz(reshape(entire_array,:))
end

function compress(nzm::Vector{<:NnzMatrix};as_emat=true)
  msg = """\n
  Cannot compress the given NnzMatrices, the nonzero indices and/or the full
  order number of rows do not matchone another.
  """

  @assert all([m.nonzero_idx == nzm[1].nonzero_idx for m in nzm]) msg
  @assert all([m.nrows == nzm[1].nrows for m in nzm]) msg

  array = hcat([m.array for m in nzm]...)

  if as_emat
    array = convert(EMatrix{Float},array)
  end

  nzm.array = array
  nzm
end

function compress(nzm::Vector{Vector{<:NnzMatrix}};kwargs...)
  map(m -> hcat!(m;kwargs...),nzm)
end

function recast(nzm::NnzMatrix{<:AbstractMatrix})
  entire_array = zeros(full_size(nzm)...)
  entire_array[nzm.nonzero_idx,:] = nzm.array
  entire_array
end

function recast(nzm::NnzMatrix{<:SparseMatrixCSC})
  sparse_rows,sparse_cols = from_vec_to_mat_idx(nzm.nonzero_idx,nzm.nrows)
  ncols = maximum(sparse_cols)
  sparse(sparse_rows,sparse_cols,nzm.array,nzm.nrows,ncols)
end

function change_mode!(nzm::NnzMatrix,sizes...)
  mode1_ndofs,mode2_ndofs,nparams = sizes

  mode2 = reshape(similar(nzm.array),mode2_ndofs,mode1_ndofs*nparams)
  _mode2(k::Int) = nzm.array[:,(k-1)*Nt+1:k*Nt]'
  @inbounds for k = 1:nparams
    setindex!(mode2,_mode2(k),:,(k-1)*mode1_ndofs+1:k*mode1_ndofs)
  end

  nzm.array = mode2
end

_compress_rows(nzm::NnzMatrix) = size(nzm.array,1) > size(nzm.array,2)

function tpod!(nzm::NnzMatrix;kwargs...)
  compress_rows = _compress_rows(nzm)
  nzm.array = tpod(Val{compress_rows}(),nzm.array;kwargs...)
end

function tpod(::Val{true},array::AbstractMatrix;ϵ=1e-4)
  compressed_array = array'*array
  _,Σ2,V = svd(compressed_array)
  Σ = sqrt.(Σ2)
  n = truncation(Σ,ϵ)
  U = array*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (vals[i]+eps())
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
