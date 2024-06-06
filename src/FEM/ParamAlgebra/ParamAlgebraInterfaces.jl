function Algebra.allocate_vector(::Type{V},n::Integer) where V<:AbstractParamContainer
  vector = zeros(eltype(V),n)
  array_of_similar_arrays(vector,param_length(V))
end

function Algebra.allocate_in_range(matrix::AbstractParamMatrix)
  V = VectorOfVectors{T,L}
  allocate_in_range(V,matrix)
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
          vij = ParamContainer(map(x->x[li,lj],param_data(vs)))
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs::AbstractParamContainer,is,js)
  @check param_length(A) == param_length(vs)
  @inbounds for i = param_eachindex(A)
    Algebra._add_entries!(combine,param_getindex(A,i),param_getindex(vs,i),is,js)
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs::Nothing,is,js)
  @inbounds for i = param_eachindex(A)
    Algebra._add_entries!(combine,param_getindex(A,i),vs,is,js)
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A,vs::AbstractParamContainer,is)
  for (li,i) in enumerate(is)
    if i>0
      vi = ParamContainer(map(x->x[li],param_data(vs)))
      add_entry!(combine,A,vi,i)
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs::Nothing,is)
  @inbounds for i = param_eachindex(A)
    Algebra._add_entries!(combine,param_getindex(A,i),vs,is)
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs::AbstractParamContainer,is)
  @check param_length(A) == param_length(vs)
  @inbounds for i = param_eachindex(A)
    Algebra._add_entries!(combine,param_getindex(A,i),param_getindex(vs,i),is)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamVector,v::Number,is)
  @inbounds for i = param_eachindex(A)
    Algebra._add_entries!(combine,param_getindex(A,i),v,is)
  end
  A
end

function Algebra.is_entry_stored(::Type{T},i,j) where T<:AbstractParamMatrix
  is_entry_stored(eltype(T),i,j)
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamMatrix,v::Number,is,js)
  @inbounds for i = param_eachindex(A)
    Algebra._add_entries!(combine,param_getindex(A,i),v,is,js)
  end
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
    @inbounds for j = param_eachindex(T)
      param_getindex(a.V,j)[k] = param_getindex(v,j)
    end
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
  ParamArray([Algebra.sparse_from_coo(eltype(T),I,J,param_getindex(V,i),m,n) for i = param_eachindex(V)])
end

function Algebra.finalize_coo!(::Type{T},I,J,V::AbstractParamArray,m,n) where T<:AbstractParamMatrix
  @inbounds for i = param_eachindex(V)
    Algebra.finalize_coo!(eltype(T),I,J,param_getindex(V,i),m,n)
  end
end

function Algebra.nz_index(a::AbstractParamMatrix,i0,i1)
  nz_index(testitem(a),i0,i1)
end

function Algebra.push_coo!(::Type{T},I,J,V::AbstractParamArray,i,j,v) where T<:AbstractParamMatrix
  @inbounds for k = param_eachindex(V)
    Algebra.push_coo!(eltype(T),I,J,param_getindex(V,k),i,j,v)
  end
end

struct ParamCounter{C,L}
  counter::C
  function ParamCounter(counter::C,::Val{L}) where {C,L}
    new{C,L}(counter)
  end
end

function ParamCounter(counter::C,L::Integer) where C
  ParamCounter(counter,Val(L))
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
  ParamInserterCSC(inserter,Val(L))
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
  ::typeof(+),a::ParamInserterCSC{Tv,Ti,P},v::AbstractParamContainer,i,j) where {Tv,Ti,P}
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for l = param_eachindex(P)
      param_getindex(a.nzval,l)[p] = param_getindex(v,l)
    end
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check  pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
      @inbounds for l = param_eachindex(P)
        param_getindex(a.nzval,l)[o] = param_getindex(a.nzval,l)[k]
      end
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for l = param_eachindex(P)
      param_getindex(a.nzval,l)[p] = param_getindex(v,l)
    end
  else
    # update existing entry
    @inbounds for l = param_eachindex(P)
      param_getindex(a.nzval,l)[p] += param_getindex(v,l)
    end
  end
  nothing
end

function Algebra.create_from_nz(a::ParamInserterCSC{Tv,Ti,P}) where {Tv,Ti,P}
  k = 1
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      @inbounds for l = param_eachindex(P)
        param_getindex(a.nzval,l)[k] = param_getindex(a.nzval,l)[p]
      end
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
