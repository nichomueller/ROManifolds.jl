eltype2(x) = eltype(eltype(x))

function Algebra.allocate_vector(::Type{V},n::Integer) where V<:AbstractParamContainer
  vector = zeros(eltype2(V),n)
  array_of_zero_arrays(vector,param_length(V))
end

function Algebra.allocate_vector(::Type{<:BlockVectorOfVectors{T,L}},indices::BlockedUnitRange) where {T,L}
  V = VectorOfVectors{T,L}
  mortar(map(ids -> allocate_vector(V,ids),blocks(indices)))
end

function Algebra.allocate_in_range(::Type{V},matrix) where V<:AbstractParamContainer
  rows = Base.OneTo(innersize(matrix,1))
  allocate_vector(V,rows)
end

function Algebra.allocate_in_range(matrix::AbstractParamMatrix)
  V = VectorOfVectors{T,L}
  allocate_in_range(V,matrix)
end

function Algebra.allocate_in_range(matrix::BlockMatrixOfMatrices{T,L}) where {T,L}
  V = BlockVectorOfVectors{T,L}
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

function Algebra.allocate_in_domain(matrix::BlockMatrixOfMatrices{T,L}) where {T,L}
  V = BlockVectorOfVectors{T,L}
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
  @inbounds for k = param_eachindex(A)
    A.data[k][i] = combine(A.data[k][i],v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamVector,v::AbstractArray,i)
  @inbounds for k = param_eachindex(A)
    A.data[k][i] = combine(A.data[k][i],v[k])
  end
  A
end

function Algebra.is_entry_stored(::Type{T},i,j) where T<:AbstractParamMatrix
  is_entry_stored(eltype(T),i,j)
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamMatrix,v::Number,i,j)
  @inbounds for k = param_eachindex(A)
    A.data[k][i,j] = combine(A.data[k][i,j],v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamMatrix,v::AbstractArray,i,j)
  @inbounds for k = param_eachindex(A)
    A.data[k][i,j] = combine(A.data[k][i,j],v[k])
  end
  A
end

function Algebra.add_entry!(combine::Function,A::MatrixOfSparseMatricesCSC,v::Number,i,j)
  k = nz_index(A,i,j)
  nz = nonzeros(A)
  @inbounds for l = param_eachindex(A)
    Aijl = nz[k,l]
    nz[k,l] = combine(Aijl,v)
  end
  A
end

function Algebra.add_entry!(combine::Function,A::MatrixOfSparseMatricesCSC,v::AbstractArray,i,j)
  k = nz_index(A,i,j)
  nz = nonzeros(A)
  @inbounds for l = param_eachindex(A)
    Aijl = nz[k,l]
    vl = v[l]
    nz[k,l] = combine(Aijl,vl)
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
    @inbounds for l = param_eachindex(T)
      a.V.data[l][k] = v.data[l]
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
  param_array(param_data(V)) do v
    Algebra.sparse_from_coo(eltype(T),I,J,v,m,n)
  end
end

function Algebra.finalize_coo!(::Type{T},I,J,V::AbstractParamArray,m,n) where T<:AbstractParamMatrix
  @inbounds for i = param_eachindex(V)
    vi = param_getindex(V,i)
    Algebra.finalize_coo!(eltype(T),I,J,vi,m,n)
  end
end

function Algebra.nz_index(A::MatrixOfSparseMatricesCSC,i0,i1)
  if !(1 <= i0 <= innersize(A,1) && 1 <= i1 <= innersize(A,2)); throw(BoundsError()); end
  ptrs = SparseArrays.getcolptr(A)
  r1 = Int(ptrs[i1])
  r2 = Int(ptrs[i1+1]-1)
  (r1 > r2) && return -1
  r1 = searchsortedfirst(rowvals(A),i0,r1,r2,Base.Order.Forward)
  ((r1 > r2) || (rowvals(A)[r1] != i0)) ? -1 : r1
end

function Algebra.push_coo!(::Type{T},I,J,V::AbstractParamArray,i,j,v) where T<:AbstractParamMatrix
  @inbounds for k = param_eachindex(V)
    vk = param_getindex(V,k)
    Algebra.push_coo!(eltype(T),I,J,vk,i,j,v)
  end
end

"""
    ParamCounter{C,L}

Extends the concept of `counter` in Gridap to accommodate a parametric setting.
L represents the parametric length of the quantity to be assembled.

"""
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
  v = similar(S,map(length,a.axes))
  array_of_zero_arrays(v,param_length(T))
end

function Algebra.nz_allocation(a::ParamCounter{C,L}) where {C,L}
  inserter = nz_allocation(a.counter)
  ParamInserter(inserter,L)
end

function ParamInserter(inserter,L::Integer)
  @notimplemented "Only implemented the CSC format"
end

function ParamInserter(inserter::Algebra.InserterCSC,L::Integer)
  ParamInserterCSC(inserter,Val{L}())
end

"""
    struct ParamInserterCSC{Tv,Ti} end

Extends the concept of `InserterCSC` in Gridap to accommodate a parametric setting.
Tv is the type of the parametric nonzero entries of the CSC matrices to be
assembled.

"""
struct ParamInserterCSC{Tv,Ti}
  nrows::Int
  ncols::Int
  colptr::Vector{Ti}
  colnnz::Vector{Ti}
  rowval::Vector{Ti}
  nzval::Tv
  function ParamInserterCSC(inserter::Algebra.InserterCSC{Tv′,Ti},::Val{L}) where {Tv′,Ti,L}
    @unpack nrows,ncols,colptr,colnnz,rowval,nzval = inserter
    pnzval = array_of_similar_arrays(nzval,L)
    Tv = typeof(pnzval)
    new{Tv,Ti}(nrows,ncols,colptr,colnnz,rowval,pnzval)
  end
end

Algebra.LoopStyle(::Type{<:ParamInserterCSC}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::ParamInserterCSC,v::Nothing,i,j)
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
  ::typeof(+),a::ParamInserterCSC{Tv},v::Number,i,j) where Tv
  @notimplemented
end

@noinline function Algebra.add_entry!(
  ::typeof(+),a::ParamInserterCSC{Tv},v::AbstractArray,i,j) where Tv
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for l = param_eachindex(Tv)
      a.nzval.data[l][p] = v[l]
    end
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check  pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
      @inbounds for l = param_eachindex(Tv)
        a.nzval[l][o] = a.nzval[l][k]
      end
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for l = param_eachindex(Tv)
      a.nzval[l][p] = v[l]
    end
  else
    # update existing entry
    @inbounds for l = param_eachindex(Tv)
      a.nzval[l][p] += v[l]
    end
  end
  nothing
end

function Algebra.create_from_nz(a::ParamInserterCSC{Tv}) where Tv
  k = 1
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      @inbounds for l = param_eachindex(Tv)
        a.nzval[l][k] = a.nzval[l][p]
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
    resize!(v,nnz)
    SparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,v)
  end
end
