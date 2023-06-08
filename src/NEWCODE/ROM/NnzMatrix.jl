mutable struct NnzMatrix{T}
  array::AbstractMatrix
  nonzero_idx::Vector{Int}
  nrows::Int
end

function NnzMatrix(entire_array::T) where {T<:AbstractMatrix}
  array,nonzero_idx = compress(entire_array)
  nrows = size(entire_array,1)
  NnzMatrix{T}(array,nonzero_idx,nrows)
end

Base.size(nzm::NnzMatrix,idx...) = size(nzm.array,idx...)

full_size(nzm::NnzMatrix) = (nzm.nrows,size(nzm,2))

Base.eltype(::Type{<:NnzMatrix{T}}) where T = T

Base.eltype(::NnzMatrix{T}) where T = T

Base.getindex(nzm::NnzMatrix,idx...) = nzm.array[idx...]

Base.eachindex(nzm::NnzMatrix) = eachindex(nzm.array)

Base.setindex!(nzm::NnzMatrix,val,idx...) = setindex!(nzm.array,val,idx...)

# function Base.setindex!(nzm::Vector{NnzMatrix{T}},val,idx...) where T
#   map((m,v) -> setindex!(m,v,idx...),nzm.array,val)
# end

function Base.copy(nzm::NnzMatrix{T}) where T
  NnzMatrix{T}(copy(nzm.array),copy(nzm.nonzero_idx),copy(nzm.nrows))
end

Base.copyto!(nzm::NnzMatrix,val::AbstractMatrix) = copyto!(nzm.array,val)

# function Base.copyto!(nzm::Vector{NnzMatrix{T}},val) where T
#   @assert length(nzm) == length(val)
#   copyto!.(nzm,val)
# end

function Base.show(io::IO,nmz::NnzMatrix{T}) where T
  print(io,"NnzMatrix{$T} storing $(length(nmz.nonzero_idx)) nonzero entries")
end

function Base.:(*)(nzm1::NnzMatrix{T},nzm2::NnzMatrix{T}) where T
  msg = """\n
  Cannot multiply the given NnzMatrices, the nonzero indices and/or the full
  order number of rows do not match one another.
  """
  @assert nzm1.nonzero_idx == nzm2.nonzero_idx msg
  @assert nzm1.nrows == nzm2.nrows msg
  array = nzm1.array*nzm2.array
  NnzMatrix{T}(array,copy(nzm1.nonzero_idx),copy(nzm1.nrows))
end

function Base.adjoint(nzm::NnzMatrix{T}) where T
  array = nzm.array'
  NnzMatrix{T}(array,copy(nzm.nonzero_idx),copy(nzm.nrows))
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

function compress(nzm::Vector{NnzMatrix{T}};as_emat=true) where T
  msg = """\n
  Cannot compress the given NnzMatrices, the nonzero indices and/or the full
  order number of rows do not match one another.
  """

  test_nnz_idx,test_nrows = nzm[1].nonzero_idx,nzm[1].nrows
  @assert all([m.nonzero_idx == test_nnz_idx for m in nzm]) msg
  @assert all([m.nrows == test_nrows for m in nzm]) msg

  array = hcat([m.array for m in nzm]...)

  if as_emat
    array = convert(EMatrix{Float},array)
  end

  NnzMatrix{T}(array,test_nnz_idx,test_nrows)
end

function compress(nzm::Vector{Vector{NnzMatrix{T}}};kwargs...) where T
  sorted_nzm(i) = map(m->getindex(m,i),nzm)
  map(i -> compress(sorted_nzm(i);kwargs...),eachindex(nzm))
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

function change_mode!(nzm::NnzMatrix,nparams::Int)
  mode1_ndofs = size(nzm,1)
  mode2_ndofs = Int(size(nzm,2)/nparams)

  mode2 = reshape(similar(nzm.array),mode2_ndofs,mode1_ndofs*nparams)
  _mode2(k::Int) = nzm.array[:,(k-1)*mode2_ndofs+1:k*mode2_ndofs]'
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
