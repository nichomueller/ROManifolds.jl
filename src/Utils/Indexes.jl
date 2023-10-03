"""Given a vector 'idx' referred to the entries of a vector 'vec' of length Ns^2,
 this function computes the row-column indexes of the NsxNs matrix associated to 'vec'"""
function vec_to_mat_idx(idx::Vector{Int},Ns::Int)
  col_idx = 1 .+ Int.(floor.((idx.-1)/Ns))
  row_idx = idx - (col_idx.-1)*Ns
  row_idx,col_idx
end

function sparse_to_full_idx(sparse_idx::Int,idx::Vector{Int})
  first(findall(x -> x == sparse_idx,idx))
end

function sparse_to_full_idx(sparse_idx,idx::Vector{Int})
  map(x->sparse_to_full_idx(x,idx),sparse_idx)
end

function spacetime_idx(
  space_idx::Vector{Int},
  time_idx::Vector{Int},
  Ns=maximum(space_idx))

  (time_idx .- 1)*Ns .+ space_idx
end

function fast_idx(kst::Int,ns::Int)
  ks = mod(kst,ns)
  ks == 0 ? ns : ks
end

function slow_idx(kst::Int,ns::Int)
  Int(floor((kst-1)/ns)+1)
end

function index_pairs(a,b)
  collect(Iterators.product(1:a,1:b))
end

param_time_idx(times::Vector,nparams::Int) = vcat((collect(1:nparams) .+ (times .- 1)'*nparams)...)
param_time_idx(ntimes::Int,nparams::Int) = param_time_idx(collect(1:ntimes),nparams)

function change_mode(mat::Matrix{T},time_ndofs::Int,nparams::Int) where T
  space_ndofs = Int(length(mat)/(time_ndofs*nparams))
  mode2 = zeros(T,time_ndofs,space_ndofs*nparams)
  @inbounds for i = 1:time_ndofs
    mode2[i,:] = reshape(mat[:,(i-1)*nparams+1:i*nparams],:)
  end
  return mode2
end

reorder_col_idx(ntimes::Int,range::UnitRange) = collect(range) .+ collect(0:ntimes-1)'*nparams
reorder_col_idx(ntimes::Int,nparams::Int) = collect(1:nparams) .+ collect(0:ntimes-1)'*nparams

function change_order(mat::Matrix{T},time_ndofs::Int) where T
  nparams = Int(size(mat,2)/time_ndofs)
  idx = reorder_col_idx(time_ndofs,nparams)
  _mat = zeros(T,size(mat))
  @inbounds for i = 1:nparams
    _mat[:,(i-1)*time_ndofs+1:i*time_ndofs] = mat[:,idx[i,:]]
  end
  return _mat
end

function idx_batches(v::AbstractArray)
  nbatches = Threads.nthreads()
  [round(Int,i) for i in range(0,length(v),nbatches+1)]
end

function idx_batches_for_id(v::AbstractArray)
  id = Threads.threadid()
  idx = idx_batches(v)
  idx[id]+1:idx[id+1]
end

function Base.argmax(v::Vector,nval::Int)
  s = sort(v,rev=true)
  Int.(indexin(s,v))[1:nval]
end
