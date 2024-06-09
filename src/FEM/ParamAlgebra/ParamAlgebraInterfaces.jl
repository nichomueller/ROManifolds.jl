eltype2(x) = eltype(eltype(x))

function Algebra.allocate_vector(::Type{V},n::Integer) where V<:AbstractParamContainer
  vector = zeros(eltype2(V),n)
  array_of_similar_arrays(vector,param_length(V))
end

function Algebra.allocate_in_range(::Type{V},matrix) where V<:AbstractParamContainer
  rows = Base.OneTo(innersize(matrix,1))
  allocate_vector(V,rows)
end

function Algebra.allocate_in_range(matrix::AbstractParamMatrix)
  V = VectorOfVectors{T,L}
  allocate_in_range(V,matrix)
end

function Algebra.allocate_in_domain(::Type{V},matrix) where V<:AbstractParamContainer
  cols = Base.OneTo(innersize(matrix,2))
  allocate_vector(V,cols)
end

function Algebra.allocate_in_domain(matrix::AbstractParamMatrix)
  V = VectorOfVectors{T,L}
  allocate_in_domain(V,matrix)
end

@inline function Algebra._add_entries!(
  combine::Function,A,vs::AbstractParamContainer,is,js)

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          vij = param_entry(vs,li,lj)
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A,vs::AbstractParamContainer,is)
  for (li,i) in enumerate(is)
    if i>0
      vi = param_entry(vs,li)
      add_entry!(combine,A,vi,i)
    end
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamVector,v::Number,i)
  V = ParamNumber(fill(v,param_length(A)))
  add_entry!(combine,A,V,i)
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamVector,v::ParamNumber,i)
  Ai = A.data[i,:]
  @inbounds A.data[i,:] .= combine(Ai,v)
  A
end

function Algebra.is_entry_stored(::Type{T},i,j) where T<:AbstractParamMatrix
  is_entry_stored(eltype(T),i,j)
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamMatrix,v::Number,i,j)
  V = ParamNumber(fill(v,param_length(A)))
  add_entry!(combine,A,V,i,j)
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamMatrix,v::ParamNumber,i,j)
  Aij = A.data[i,j,:]
  @inbounds A.data[i,j,:] .= combine(Aij,v)
  A
end

function Algebra.add_entry!(combine::Function,A::MatrixOfSparseMatricesCSC,v::ParamNumber,i,j)
  k = nz_index(A,i,j)
  Aij = A.data[k,:]
  @inbounds A.data[k,:] .= combine(Aij,v)
  A
end

@inline function Algebra.add_entry!(::typeof(+),a::Algebra.AllocationCOO{T},::Nothing,i,j) where T<:AbstractParamMatrix
  if Algebra.is_entry_stored(T,i,j)
    a.counter.nnz = a.counter.nnz + 1
    k = a.counter.nnz
    a.I[k] = i
    a.J[k] = j
  end
  nothing
end

@inline function Algebra.add_entry!(::typeof(+),a::Algebra.AllocationCOO{T},v,i,j) where T<:AbstractParamMatrix
  if Algebra.is_entry_stored(T,i,j)
    a.counter.nnz = a.counter.nnz + 1
    k = a.counter.nnz
    a.I[k] = i
    a.J[k] = j
    a.V.data[k,j] .= v.data[:,j]
  end
  nothing
end

function Algebra.allocate_coo_vectors(::Type{T},n::Integer) where {Tv,Ti,T<:MatrixOfSparseMatricesCSC{Tv,Ti}}
  I = zeros(Ti,n)
  J = zeros(Ti,n)
  V = zeros(Tv,n)
  PV = array_of_similar_arrays(V,param_length(T))
  I,J,PV
end

function Algebra.sparse_from_coo(::Type{T},I,J,V::AbstractParamArray,m,n) where T<:AbstractParamMatrix
  param_array(param_data(V)) do v
    cv = collect(v)
    Algebra.sparse_from_coo(eltype(T),I,J,cv,m,n)
  end
end

function Algebra.finalize_coo!(::Type{T},I,J,V::AbstractParamArray,m,n) where T<:AbstractParamMatrix
  @inbounds for i = param_eachindex(V)
    vi = param_view(V,i)
    Algebra.finalize_coo!(eltype(T),I,J,vi,m,n)
  end
end

function Algebra.nz_index(a::AbstractParamMatrix,i0,i1)
  nz_index(testitem(a),i0,i1)
end

function Algebra.push_coo!(::Type{T},I,J,V::AbstractParamArray,i,j,v) where T<:AbstractParamMatrix
  @inbounds for k = param_eachindex(V)
    vk = param_view(V,k)
    Algebra.push_coo!(eltype(T),I,J,vk,i,j,v)
  end
end

struct ParamCounter{C,L}
  counter::C
  function ParamCounter(counter::C,::Val{L}) where {C,L}
    new{C,L}(counter)
  end
end

function ParamCounter(counter::C,L::Integer) where C
  ParamCounter(counter,Val{L}())
end

Algebra.LoopStyle(::Type{ParamCounter{C,L}}) where {C,L} = LoopStyle(C)

@inline function Algebra.add_entry!(::typeof(+),a::ParamCounter,v,i,j)
  add_entry!(+,a.counter,v,i,j)
end

function Algebra.nz_counter(builder::SparseMatrixBuilder{T},axes) where T<:AbstractParamMatrix
  counter = nz_counter(SparseMatrixBuilder(eltype(T)),axes)
  ParamCounter(counter,param_length(T))
end

function Algebra.nz_allocation(a::Algebra.ArrayCounter{T}) where T<:AbstractParamVector
  S = eltype(T)
  v = fill!(similar(S,map(length,a.axes)),zero(eltype(S)))
  array_of_similar_arrays(v,param_length(T))
end

function Algebra.nz_allocation(a::ParamCounter{C,L}) where {C,L}
  inserter = nz_allocation(a.counter)
  ParamInserter(inserter,L)
end

# CSC format

function ParamInserter(inserter::Algebra.InserterCSC,L::Integer)
  ParamInserterCSC(inserter,Val{L}())
end

struct ParamInserterCSC{Tv,Ti,P}
  nrows::Int
  ncols::Int
  colptr::Vector{Ti}
  colnnz::Vector{Ti}
  rowval::Vector{Ti}
  nzval::P
  function ParamInserterCSC(inserter::Algebra.InserterCSC{Tv,Ti},::Val{L}) where {Tv,Ti,L}
    @unpack nrows,ncols,colptr,colnnz,rowval,nzval = inserter
    pnzval = array_of_similar_arrays(nzval,L)
    P = typeof(pnzval)
    new{Tv,Ti,P}(nrows,ncols,colptr,colnnz,rowval,pnzval)
  end
end

Algebra.LoopStyle(::Type{<:ParamInserterCSC}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::ParamInserterCSC{Tv,Ti},v::Nothing,i,j)  where {Tv,Ti}
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check  pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
  end
  nothing
end

@noinline function Algebra.add_entry!(
  ::typeof(+),a::ParamInserterCSC{Tv,Ti,P},v::Union{ParamNumber,Number},i,j) where {Tv,Ti,P}
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    a.nzval.data[p,:] .= v
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check  pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
      a.nzval.data[o,:] .= a.nzval.data[k,:]
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    a.nzval.data[p,:] .= v
  else
    # update existing entry
    a.nzval.data[p,:] .+= v
  end
  nothing
end

function Algebra.create_from_nz(a::ParamInserterCSC{Tv,Ti,P}) where {Tv,Ti,P}
  k = 1
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      a.nzval.data[k,:] .= a.nzval.data[p,:]
      a.rowval[k] = a.rowval[p]
      k += 1
    end
  end
  @inbounds for j in 1:a.ncols
    a.colptr[j+1] = a.colnnz[j]
  end
  length_to_ptrs!(a.colptr)
  nnz = a.colptr[end]-1
  resize!(a.rowval,nnz)

  param_array(param_data(a.nzval)) do v
    cv = collect(v)
    resize!(cv,nnz)
    SparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,cv)
  end
end
