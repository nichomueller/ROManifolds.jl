function Algebra.allocate_vector(::Type{V},n::Integer) where V<:ParamTTVector
  vector = zeros(eltype(V),n)
  ttvector = TTArray(vector,Val(get_dim(V)))
  allocate_param_array(ttvector,length(V))
end

function FESpaces.SparseMatrixAssembler(
  trial::SingleFieldFESpace,
  test::UnconstrainedFESpace{<:TTVector{D}}) where D

  T = get_dof_value_type(trial)
  matrix_type = TTSparseMatrixCSC{D,T}
  vector_type = TTVector{D,T}
  SparseMatrixAssembler(matrix_type,vector_type,trial,test)
end

function Algebra.allocate_in_range(matrix::TTMatrix{D,T}) where {D,T}
  V = TTVector{D,T}
  allocate_in_range(V,matrix)
end

function Algebra.allocate_in_domain(matrix::TTMatrix{D,T}) where {D,T}
  V = TTVector{D,T}
  allocate_in_domain(V,matrix)
end

function Algebra.is_entry_stored(::Type{TTSparseMatrix{D,T,V}},i,j) where {D,T,V}
  is_entry_stored(V,i,j)
end

@inline function Algebra.add_entry!(combine::Function,A::TTArray,v::Number,i,j)
  add_entry!(combine,A.values,v,i,j)
  A
end

@inline function Algebra.add_entry!(combine::Function,A::TTArray,::Nothing,i,j)
  add_entry!(combine,A.values,nothing,i,j)
  A
end

function Algebra.allocate_coo_vectors(::Type{TTSparseMatrix{D,T,V}},n::Integer) where {D,T,V}
  TTArray(Algebra.allocate_coo_vectors(V,n),Val(D))
end

function Algebra.sparse_from_coo(::Type{TTSparseMatrix{D,T,V}},i,j,v,m,n) where {D,T,V}
  TTArray(Algebra.sparse_from_coo(V,i,j,v,m,n),Val(D))
end

function Algebra.finalize_coo!(::Type{TTSparseMatrix{D,T,V}},i,j,v,m,n) where {D,T,V}
  Algebra.finalize_coo!(V,i,j,v,m,n)
end

function Algebra.nz_index(a::TTSparseMatrix,i0,i1)
  Algebra.nz_index(a.values,i0,i1)
end

function Algebra.push_coo!(::Type{TTSparseMatrix{D,T,W}},I,J,V::ParamArray,i,j,v) where {D,T,W}
  Algebra.push_coo!(W,i,j,v,m,n)
end

struct TTCounter{C,D}
  counter::C
  function TTCounter(counter::C,::Val{D}) where {C,D}
    new{C,D}(counter)
  end
end

Algebra.LoopStyle(::Type{TTCounter{C,D}}) where {C,D} = LoopStyle(C)

@inline function Algebra.add_entry!(::typeof(+),a::TTCounter,v,i,j)
  add_entry!(+,a.counter,v,i,j)
end

function Algebra.nz_counter(builder::SparseMatrixBuilder{<:TTSparseMatrixCSC{D,T}},axes) where {D,T}
  ttbuilder = SparseMatrixBuilder(SparseMatrixCSC{T,Int})
  counter = nz_counter(ttbuilder,axes)
  TTCounter(counter,Val(D))
end

function Algebra.nz_allocation(a::TTCounter{C,D}) where {C,D}
  inserter = nz_allocation(a.counter)
  TTInserter(inserter,Val(D))
end

struct TTInserter{I,D}
  inserter::I
  function TTInserter(inserter::I,::Val{D}) where {I,D}
    new{I,D}(inserter)
  end
end

Algebra.LoopStyle(::Type{<:TTInserter}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::TTInserter,v::Nothing,i,j)
  Algebra.add_entry!(+,a.inserter,v,i,j)
end

@noinline function Algebra.add_entry!(::typeof(+),a::TTInserter,v,i,j)
  Algebra.add_entry!(+,a.inserter,v,i,j)
end

function Algebra.create_from_nz(a::TTInserter{I,D}) where {I,D}
  nz = create_from_nz(a.inserter)
  TTArray(nz,Val(D))
end

# Param CSC

function ParamInserter(inserter::TTInserter,L::Integer)
  ParamTTInserterCSC(inserter,Val(L))
end

struct ParamTTInserterCSC{Tv,Ti,P,D}
  nrows::Int
  ncols::Int
  colptr::Vector{Ti}
  colnnz::Vector{Ti}
  rowval::Vector{Ti}
  nzval::P
  function ParamTTInserterCSC(a::TTInserter{Algebra.InserterCSC{Tv,Ti},D},::Val{L}) where {Tv,Ti,D,L}
    @unpack nrows,ncols,colptr,colnnz,rowval,nzval = a.inserter
    pnzval = allocate_param_array(nzval,L)
    P = typeof(pnzval)
    new{Tv,Ti,P,D}(nrows,ncols,colptr,colnnz,rowval,pnzval)
  end
end

Algebra.LoopStyle(::Type{<:ParamTTInserterCSC}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::ParamTTInserterCSC{Tv,Ti},v::Nothing,i,j)  where {Tv,Ti}
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
  ::typeof(+),a::ParamTTInserterCSC{Tv,Ti,P},v::AbstractParamContainer{Tv},i,j) where {Tv,Ti,P}
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

function Algebra.create_from_nz(a::ParamTTInserterCSC{Tv,Ti,P,D}) where {Tv,Ti,P,D}
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
    v = SparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,a.nzval[l])
    TTArray(v,Val(D))
  end
  ParamArray(csc)
end
