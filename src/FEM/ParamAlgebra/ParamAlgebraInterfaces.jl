function allocate_param_vector(::Type{V},indices,plength) where V
  n = length(indices)
  allocate_param_vector(V,n,plength)
end

function allocate_param_vector(::Type{V},n::Integer,plength::Integer) where V<:ParamVector
  array_of_similar_arrays(cache,param_length(mat))
end

@inline function Algebra._add_entries!(
  combine::Function,A,vs::AbstractParamContainer,is,js)
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          vij = ParamContainer(map(x->x[li,lj],vs))
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs::AbstractParamContainer,is,js)
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          for k = eachindex(vs)
            vij = vs[k][li,lj]
            add_entry!(combine,A[k],vij,i,j)
          end
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
      vi = ParamContainer(map(x->x[li],vs))
      add_entry!(combine,A,vi,i)
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs::AbstractParamContainer,is)
  for (li,i) in enumerate(is)
    if i>0
      for k = eachindex(vs)
        vi = vs[k][li]
        add_entry!(combine,A[k],vi,i)
      end
    end
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ParamVector,v::Number,i)
  for k = eachindex(A)
    ai = A[k][i]
    A[k][i] = combine(ai,v)
  end
  A
end

function Algebra.is_entry_stored(::Type{<:ParamMatrix{T,L,A}},i,j) where {T,L,A}
  is_entry_stored(eltype(A),i,j)
end

@inline function Algebra.add_entry!(combine::Function,A::ParamMatrix,v::Number,i,j)
  for k = eachindex(A)
    aij = A[k][i,j]
    A[k][i,j] = combine(aij,v)
  end
  A
end

@inline function Algebra.add_entry!(::typeof(+),a::Algebra.AllocationCOO{T},::Nothing,i,j) where T<:ParamMatrix
  if Algebra.is_entry_stored(T,i,j)
    a.counter.nnz = a.counter.nnz + 1
    k = a.counter.nnz
    a.I[k] = i
    a.J[k] = j
  end
  nothing
end

@inline function Algebra.add_entry!(::typeof(+),a::Algebra.AllocationCOO{T},v,i,j) where T<:ParamMatrix
  if Algebra.is_entry_stored(T,i,j)
    a.counter.nnz = a.counter.nnz + 1
    k = a.counter.nnz
    a.I[k] = i
    a.J[k] = j
    for l = 1:length(T)
      a.V[l][k] = v[l]
    end
  end
  nothing
end

function Algebra.allocate_coo_vectors(
  ::Type{<:ParamMatrix{Tv,L,<:Vector{<:AbstractSparseMatrix{Tv,Ti}}}},
  n::Integer) where {Tv,Ti,L}
  I = zeros(Ti,n)
  J = zeros(Ti,n)
  V = zeros(Tv,n)
  PV = allocate_param_array(V,L)
  I,J,PV
end

function Algebra.sparse_from_coo(::Type{ParamMatrix{T,L,A}},I,J,V::ParamArray,m,n) where {T,L,A}
  elA = eltype(A)
  psparse = map(1:L) do k
    Algebra.sparse_from_coo(elA,I,J,V[k],m,n)
  end
  ParamArray(psparse)
end

function Algebra.finalize_coo!(::Type{ParamMatrix{T,L,A}},I,J,V::ParamArray,m,n) where {T,L,A}
  elA = eltype(A)
  map(1:L) do k
    Algebra.finalize_coo!(elA,I,J,V[k],m,n)
  end
end

function Algebra.nz_index(a::ParamMatrix,i0,i1)
  nz_index(first(a),i0,i1)
end

function Algebra.push_coo!(::Type{ParamMatrix{T,L,A}},I,J,V::ParamArray,i,j,v) where {T,L,A}
  elA = eltype(A)
  map(1:L) do k
    Algebra.push_coo!(elA,I,J,V[k],i,j,v)
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

function Algebra.nz_counter(
  builder::SparseMatrixBuilder{<:ParamMatrix{T,L,A}},
  axes) where {T,L,A}

  elA = eltype(A)
  elb = SparseMatrixBuilder(elA)
  counter = nz_counter(elb,axes)
  ParamCounter(counter,L)
end

function Algebra.nz_allocation(a::Algebra.ArrayCounter{<:ParamVector{T,L,A}}) where {T,L,A}
  elA = eltype(A)
  v = fill!(similar(elA,map(length,a.axes)),zero(T))
  allocate_param_array(v,L)
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
    pnzval = allocate_param_array(nzval,L)
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
  ::typeof(+),a::ParamInserterCSC{Tv,Ti,P},v::AbstractParamContainer{Tv},i,j) where {Tv,Ti,P}
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for l = 1:length(P)
      a.nzval[l][p] = v[l]
    end
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check  pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
      @inbounds for l = 1:length(P)
        a.nzval[l][o] = a.nzval[l][k]
      end
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for l = 1:length(P)
      a.nzval[l][p] = v[l]
    end
  else
    # update existing entry
    @inbounds for l = 1:length(P)
      a.nzval[l][p] += v[l]
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
      @inbounds for l = 1:length(P)
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
  resize!(a.nzval,nnz)
  csc = map(1:length(P)) do l
    SparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,a.nzval[l])
  end
  ParamArray(csc)
end
