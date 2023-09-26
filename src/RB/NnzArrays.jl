abstract type NnzArray{T,N} <: AbstractArray{T,N} end

Base.size(nza::NnzArray,idx...) = size(nza.nonzero_val,idx...)
Base.getindex(nza::NnzArray,idx...) = getindex(nza.nonzero_val,idx...)
Base.eachcol(nza::NnzArray) = eachcol(nza.nonzero_val)
get_nonzero_val(nza::NnzArray) = nza.nonzero_val
get_nonzero_idx(nza::NnzArray) = nza.nonzero_idx
get_nrows(nza::NnzArray) = nza.nrows

function get_nonzero_val(nza::NTuple{N,NnzArray}) where N
  hcat(map(get_nonzero_val,nza)...)
end

function get_nonzero_idx(nza::NTuple{N,NnzArray}) where N
  nz_idx = map(get_nonzero_idx,nza)
  @check all([i == first(nz_idx) for i in nz_idx])
  first(nz_idx)
end

function get_nrows(nza::NTuple{N,NnzArray}) where N
  nrows = map(get_nrows,nza)
  @check all([r == first(nrows) for r in nrows])
  first(nrows)
end

struct NnzVector{T} <: NnzArray{T,1}
  nonzero_val::Vector{T}
  nonzero_idx::Vector{Int}
  nrows::Int

  function NnzVector(
    nonzero_val::Matrix{T},
    nonzero_idx::Vector{Int},
    nrows::Int) where T

    new{T}(nonzero_val,nonzero_idx,nrows)
  end

  function NnzVector(mat::SparseMatrixCSC{T,Int}) where T
    nonzero_idx,nonzero_val = findnz(mat[:])
    nrows = size(mat,1)
    new{T}(nonzero_val,nonzero_idx,nrows)
  end
end

Base.length(nzv::NnzVector) = length(nzv.nonzero_val)

struct NnzMatrix{T} <: NnzArray{T,2}
  nonzero_val::Matrix{T}
  nonzero_idx::Vector{Int}
  nrows::Int
  nparams::Int

  function NnzMatrix(
    nonzero_val::Matrix{T},
    nonzero_idx::Vector{Int},
    nrows::Int,
    nparams::Int) where T

    new{T}(nonzero_val,nonzero_idx,nrows,nparams)
  end

  function NnzMatrix(val::AbstractArray{T}...;nparams=length(val)) where T
    vals = hcat(val...)
    nonzero_idx,nonzero_val = compress_array(vals)
    nrows = size(vals,1)
    new{T}(nonzero_val,nonzero_idx,nrows,nparams)
  end

  function NnzMatrix(val::NnzVector{T}...;nparams=length(val)) where T
    nonzero_val = get_nonzero_val(val)
    nonzero_idx = get_nonzero_idx(val)
    nrows = get_nrows(val)
    new{T}(nonzero_val,nonzero_idx,nrows,nparams)
  end

  function NnzMatrix(val::PTArray;nparams=length(val))
    NnzMatrix(get_array(val)...;nparams)
  end
end

function NnzArray(
  val::Vector{<:PTArray{Vector{T}}};
  nparams=length(testitem(val))) where T

  @check all([length(vali) == nparams for vali in val])
  NnzMatrix(get_array(hcat(val...))...;nparams)
end

Base.length(nza::NnzMatrix) = length(nza.nparams)
num_params(nzm::NnzMatrix) = nzm.nparams
num_space_dofs(nzm::NnzMatrix) = size(nzm,1)
num_time_dofs(nzm::NnzMatrix) = Int(size(nzm,2)/nzm.nparams)

function Base.copy(nzm::NnzMatrix)
  NnzMatrix(
    copy(nzm.nonzero_val),
    copy(nzm.nonzero_idx),
    copy(nzm.nrows),
    copy(nzm.nparams))
end

function Base.show(io::IO,nzm::NnzMatrix)
  print(io,"NnzMatrix storing $(length(nzm)) compressed transient snapshots")
end

function Base.prod(nza1::NnzMatrix,nza2::NnzMatrix)
  nonzero_vals = nza1.nonzero_val' * nza2.nonzero_val
  NnzMatrix(nonzero_vals,nza1.nonzero_idx,nza1.nrows,nza1.nparams)
end

function Base.prod(nzm::NnzMatrix,a::AbstractArray)
  nonzero_vals = nzm.nonzero_vals' * a
  NnzMatrix(nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function Base.prod(a::AbstractArray,nzm::NnzMatrix)
  nonzero_vals = a' * nzm.nonzero_vals
  NnzMatrix(nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function recast(nzm::NnzMatrix{T}) where T
  m = zeros(T,nzm.nrows,size(nzm,2))
  m[nzm.nonzero_idx,:] = nzm.nonzero_val
  m
end

function compress(a::AbstractMatrix,nzm::NnzMatrix{T}) where T
  m = zeros(T,nzm.nrows,size(nzm,2))
  m[nzm.nonzero_idx,:] = nzm.nonzero_val
  [a'*v for v in eachcol(m)]
end

function compress(a::AbstractMatrix,b::AbstractMatrix,nzm::NnzMatrix)
  irow,icol = from_vec_to_mat_idx(nzm.nonzero_idx,nzm.nrows)
  ncols = maximum(icol)
  map(eachcol(nzm)) do nzv
    m = sparse(irow,icol,nzv,nzm.nrows,ncols)
    a'*m*b
  end
end

function recast_idx(nzm::NnzMatrix,idx::Vector{Int})
  nonzero_idx = nzm.nonzero_idx
  nrows = nzm.nrows
  entire_idx = nonzero_idx[idx]
  entire_idx_rows,_ = from_vec_to_mat_idx(entire_idx,nrows)
  return entire_idx_rows
end

function get_at_params(range::UnitRange,nzm::NnzMatrix{T},transpose=false) where T
  space_ndofs = num_space_dofs(nzm)
  time_ndofs = num_time_dofs(nzm)
  nparams = length(range)
  idx = time_param_idx(time_ndofs,range)
  if transpose
    mat = zeros(T,time_ndofs,space_ndofs*nparams)
    @inbounds for col = eachcol(idx)
      mat[col,:] = reshape(nzm.nonzero_val[:,col]',:)
    end
  else
    mat = zeros(T,space_ndofs,time_ndofs*nparams)
    @inbounds for col = eachcol(idx)
      mat[:,col] = nzm.nonzero_val[:,col]
    end
  end
  return mat
end

function change_mode(nzm::NnzMatrix{T}) where T
  space_ndofs = num_space_dofs(nzm)
  time_ndofs = num_time_dofs(nzm)
  nparams = num_params(nzm)
  idx = time_param_idx(time_ndofs,nparams)

  mode2 = zeros(T,time_ndofs,space_ndofs*nparams)
  @inbounds for col = eachcol(idx)
    mode2[n,:] = reshape(nzm.nonzero_val[:,col]',:)
  end

  return NnzMatrix(mode2,nzm.nonzero_idx,nzm.nrows,nparams)
end

function tpod(nzm::NnzMatrix,args...;kwargs...)
  nonzero_val = tpod(nzm.nonzero_val,args...;kwargs...)
  return NnzMatrix(nonzero_val,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function collect_residuals(
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table)

  b = allocate_residual(feop,sols)
  collect_residuals!(b,fesolver,feop,sols,μ)
end

function collect_residuals!(
  b::PTArray,
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table)

  dt,θ = fesolver.dt,fesolver.θ
  dtθ = θ == 0.0 ? dt : dt*θ
  times = get_times(fesolver)

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,times)
  ode_cache = update_cache!(ode_cache,ode_op,μ,times)

  nlop = PThetaMethodNonlinearOperator(ode_op,μ,times,dtθ,sols,ode_cache,sols)
  separate_contribs = Val(true)

  printstyled("Computing fe residuals for every time and parameter\n";color=:blue)
  ress,meas = residual!(b,nlop,sols,separate_contribs)
  return NnzMatrix.(ress),meas
end

function collect_jacobian(
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table;
  kwargs...)

  A = allocate_jacobian(feop,sols)
  collect_jacobians!(A,fesolver,feop,sols,μ;kwargs...)
end

function collect_jacobian!(
  A::PTArray,
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table;
  i=1)

  dt,θ = fesolver.dt,fesolver.θ
  dtθ = θ == 0.0 ? dt : dt*θ
  times = get_times(fesolver)

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,times)
  ode_cache = update_cache!(ode_cache,ode_op,μ,times)

  nlop = PThetaMethodNonlinearOperator(ode_op,μ,times,dtθ,sols,ode_cache,sols)
  separate_contribs = Val(true)

  printstyled("Computing fe jacobian #$i for every time and parameter\n";color=:blue)
  jacs_i,meas = jacobian!(A,nlop,sols,i,separate_contribs)
  nnz_jac_i = map(x->NnzMatrix(map(NnzVector,x)),jacs_i)
  return nnz_jac_i,meas
end

struct BlockNnzMatrix{T} <: AbstractVector{NnzMatrix{T}}
  blocks::Vector{NnzMatrix{T}}

  function BlockNnzMatrix(blocks::Vector{NnzMatrix{T}}) where T
    @check all([length(nzm) == length(blocks[1]) for nzm in blocks[2:end]])
    new{T}(blocks)
  end
end

function NnzArray(val::Vector{<:PTArray{Vector{Vector{T}}}}) where T
  blocks = map(val) do vali
    array = get_array(hcat(vali...))
    NnzMatrix(array...)
  end
  BlockNnzMatrix(blocks)
end

Base.size(nzm::BlockNnzMatrix,idx...) = map(x->size(x,idx...),nzm.blocks)
Base.length(nzm::BlockNnzMatrix) = length(nzm.blocks[1])
Base.getindex(nzm::BlockNnzMatrix,idx...) = nzm.blocks[idx...]
Base.iterate(nzm::BlockNnzMatrix,args...) = iterate(nzm.blocks,args...)
get_nfields(nzm::BlockNnzMatrix) = length(nzm.blocks)
