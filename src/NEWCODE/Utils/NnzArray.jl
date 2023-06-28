abstract type AbstractNnzArray end
abstract type AbstractNnzVector <: AbstractNnzArray end
abstract type AbstractNnzMatrix <: AbstractNnzArray end

mutable struct NnzVector <: AbstractNnzVector
  nonzero_val::AbstractVector
  nonzero_idx::Vector{Int}
  nrows::Int
end

function compress(spmat::SparseMatrixCSC)
  nonzero_idx,nonzero_val = compress(spmat)
  nrows = size(spmat,1)
  NnzVector(nonzero_val,nonzero_idx,nrows)
end

mutable struct NnzMatrix <: AbstractNnzMatrix
  nonzero_val::AbstractMatrix
  nonzero_idx::Vector{Int}
  nrows::Int
end

for T in (:NnzVector,:NnzMatrix)
  @eval begin
    function Base.hcat(nzv_vec::Vector{$T}...)
      msg = """\n
      Cannot hcat the given NnzVectors: the nonzero indices and/or the full
      order number of rows do not match one another.
      """

      test_nnz_idx = nzv_vec[1].nonzero_idx
      test_nrows = nzv_vec[1].nrows
      @assert all([nzv.nonzero_idx == test_nnz_idx for nzv in nzv_vec]) msg
      @assert all([nzv.nrows == test_nrows for nzv in nzv_vec]) msg

      nzm = hcat([nzv.nonzero_val for nzv in nzv_vec]...)

      NnzMatrix(nzm,test_nnz_idx,test_nrows)
    end
  end
end

for T in (:NnzVector,:NnzMatrix)
  @eval begin
    Base.size(nz::$T,idx...) = size(nz.nonzero_val,idx...)

    Base.getindex(nz::$T,idx...) = nz.nonzero_val[idx...]

    Base.eachindex(nz::$T) = eachindex(nz.nonzero_val)

    Base.setindex!(nz::$T,val,idx...) = setindex!(nz.nonzero_val,val,idx...)

    function Base.show(io::IO,nz::$T)
      print(io,"$T storing $(length(nz.nonzero_idx)) nonzero values")
    end

    function Base.reshape(nz::$T,size...)
      rnz = reshape(nz.nonzero_val,size...)
      if isa(rnz,AbstractVector)
        NnzVector(rnz,nz.nonzero_idx,nz.nrows)
      elseif isa(rnz,AbstractMatrix)
        NnzMatrix(rnz,nz.nonzero_idx,nz.nrows)
      else
        @unreachable
      end
    end
  end
end

function Base.copy(nzv::NnzVector)
  NnzVector(copy(nzv.nonzero_val),copy(nzv.nonzero_idx),copy(nzv.nrows))
end

Base.copyto!(nzv::NnzVector,val::AbstractVector) = copyto!(nzv.nonzero_val,val)

Base.convert(::Type{Any},nzv::NnzVector) = nzv

function Base.convert(::Type{T},nzv::NnzVector) where T
  nzv_copy = copy(nzv)
  nzv_copy.nonzero_val = convert(T,nzv_copy.nonzero_val)
  nzv_copy
end

function recast(nzv::NnzVector)
  sparse_rows,sparse_cols = from_vec_to_mat_idx(nzv.nonzero_idx,nzv.nrows)
  ncols = maximum(sparse_cols)
  sparse(sparse_rows,sparse_cols,nzv.nonzero_val,nzv.nrows,ncols)
end

function Base.copy(nzm::NnzMatrix)
  NnzMatrix(copy(nzm.nonzero_val),copy(nzm.nonzero_idx),copy(nzm.nrows))
end

Base.copyto!(nzm::NnzMatrix,val::AbstractMatrix) = copyto!(nzm.nonzero_val,val)

Base.convert(::Type{Any},nzm::NnzMatrix) = nzm

function Base.convert(::Type{T},nzm::NnzMatrix) where T
  nzm_copy = copy(nzm)
  nzm_copy.nonzero_val = convert(T,nmv_copy.nonzero_val)
  nzm_copy
end

function Base.:(*)(nzm1::NnzMatrix,nzm2::NnzMatrix)
  msg = """\n
  Cannot multiply the given NnzMatrix, the nonzero indices and/or the full
  order number of rows do not match one another.
  """
  @assert nzm1.nonzero_idx == nzm2.nonzero_idx msg
  @assert nzm1.nrows == nzm2.nrows msg
  mat = nzm1.nonzero_val*nzm2.nonzero_val
  NnzMatrix(mat,copy(nzm1.nonzero_idx),copy(nzm1.nrows))
end

function Base.adjoint(nzm::NnzMatrix)
  mat = nzm.nonzero_val'
  NnzMatrix(mat,copy(nzm.nonzero_idx),copy(nzm.nrows))
end

function Gridap.FESpaces.allocate_matrix(nzm::NnzMatrix,sizes...)
  allocate_matrix(nzm.nonzero_val,sizes...)
end

function recast(nzm::NnzMatrix)
  nvec = size(nzm.nonzero_val,2)
  spm_vec = Vector{SparseMatrixCSC{Float64,Int}}(undef,nvec)
  for (ncol,col) in enumerate(eachcol(nzm.nonzero_val))
    sparse_rows,sparse_cols = from_vec_to_mat_idx(nzm.nonzero_idx,nzm.nrows)
    ncols = maximum(sparse_cols)
    spm = sparse(sparse_rows,sparse_cols,col,nzm.nrows,ncols)
    setindex!(spm_vec,spm,ncol)
  end
  spm_vec
end

function change_mode!(nzm::NnzMatrix,nparams::Int)
  mode1_ndofs = size(nzm,1)
  mode2_ndofs = Int(size(nzm,2)/nparams)

  mode2 = reshape(similar(nzm.nonzero_val),mode2_ndofs,mode1_ndofs*nparams)
  _mode2(k::Int) = nzm.nonzero_val[:,(k-1)*mode2_ndofs+1:k*mode2_ndofs]'
  @inbounds for k = 1:nparams
    setindex!(mode2,_mode2(k),:,(k-1)*mode1_ndofs+1:k*mode1_ndofs)
  end

  nzm.nonzero_val = mode2
  return
end

function change_mode(nzm::NnzMatrix,nparams::Int)
  nzm_copy = copy(nzm)
  change_mode!(nzm_copy,nparams)
  nzm_copy
end

function tpod!(nzm::NnzMatrix;kwargs...)
  nzm.nonzero_val = tpod(nzm.nonzero_val;kwargs...)
  return
end

function tpod(nzm::NnzMatrix;kwargs...)
  nzm_copy = copy(nzm)
  tpod!(nzm_copy;kwargs...)
  nzm_copy
end
