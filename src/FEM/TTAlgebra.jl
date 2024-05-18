# function FESpaces.SparseMatrixAssembler(
#   mat,
#   vec,
#   trial::T,
#   test::TProductFESpace,
#   strategy::AssemblyStrategy=DefaultAssemblyStrategy()
#   ) where T<:FESpace # add some condition on T

#   rows = get_free_dof_ids(test)
#   cols = get_free_dof_ids(trial)
#   index_map = test.dof_permutation
#   matrix_builder = TTBuilder(SparseMatrixBuilder(mat),index_map)
#   vector_builder = TTBuilder(ArrayBuilder(vec),index_map)
#   GenericSparseMatrixAssembler(
#     matrix_builder,
#     vector_builder,
#     rows,
#     cols,
#     strategy)
# end

# function Algebra.is_entry_stored(::Type{TTSparseMatrix{D,T,V}},i,j) where {D,T,V}
#   is_entry_stored(V,i,j)
# end

# @inline function Algebra.add_entry!(combine::Function,A::TTArray,v::Number,i,j)
#   add_entry!(combine,A.values,v,i,j)
#   A
# end

# @inline function Algebra.add_entry!(combine::Function,A::TTArray,::Nothing,i,j)
#   add_entry!(combine,A.values,nothing,i,j)
#   A
# end

# # function Algebra.allocate_coo_vectors(::Type{TTSparseMatrix{D,T,V}},n::Integer) where {D,T,V}
# #   TTArray(Algebra.allocate_coo_vectors(V,n),Val(D))
# # end

# # function Algebra.sparse_from_coo(::Type{TTSparseMatrix{D,T,V}},i,j,v,m,n) where {D,T,V}
# #   TTArray(Algebra.sparse_from_coo(V,i,j,v,m,n),Val(D))
# # end

# function Algebra.finalize_coo!(::Type{TTSparseMatrix{D,T,V}},i,j,v,m,n) where {D,T,V}
#   Algebra.finalize_coo!(V,i,j,v,m,n)
# end

# function Algebra.nz_index(a::TTSparseMatrix,i0,i1)
#   Algebra.nz_index(a.values,i0,i1)
# end

# function Algebra.push_coo!(::Type{TTSparseMatrix{D,T,W}},I,J,V::ParamArray,i,j,v) where {D,T,W}
#   Algebra.push_coo!(W,i,j,v,m,n)
# end

# struct TTBuilder{A,D}
#   builder::A
#   index_map::IndexMap{D}
# end

# function Algebra.get_array_type(b::TTBuilder)
#   T = get_array_type(b.builder)
#   a = T(undef,tfill(0,Val(ndims(T)))...)
#   return typeof(TTArray(a,b.index_map))
# end

# function Algebra.nz_counter(b::TTBuilder,axes)
#   counter = nz_counter(b.builder,axes)
#   TTCounter(counter,b.index_map)
# end

# struct TTCounter{C,D}
#   counter::C
#   index_map::IndexMap{D}
# end

# Algebra.LoopStyle(::Type{TTCounter{C,D}}) where {C,D} = LoopStyle(C)

# @inline function Algebra.add_entry!(::typeof(+),a::TTCounter,v,i,j)
#   add_entry!(+,a.counter,v,i,j)
# end

# function Algebra.nz_allocation(a::TTCounter{C,D}) where {C,D}
#   inserter = nz_allocation(a.counter)
#   TTInserter(inserter,a.index_map)
# end

# struct TTInserter{I,D}
#   inserter::I
#   index_map::IndexMap{D}
# end

# Algebra.LoopStyle(::Type{<:TTInserter}) = Loop()

# @inline function Algebra.add_entry!(::typeof(+),a::TTInserter,v::Nothing,i,j)
#   Algebra.add_entry!(+,a.inserter,v,i,j)
# end

# @noinline function Algebra.add_entry!(::typeof(+),a::TTInserter,v,i,j)
#   Algebra.add_entry!(+,a.inserter,v,i,j)
# end

# function Algebra.create_from_nz(a::TTInserter{I,D}) where {I,D}
#   nz = create_from_nz(a.inserter)
#   TTArray(nz,a.index_map)
# end

# # Param CSC

# function ParamInserter(inserter::TTInserter,L::Integer)
#   ParamTTInserterCSC(inserter,Val(L))
# end

# struct ParamTTInserterCSC{Tv,Ti,P,D}
#   nrows::Int
#   ncols::Int
#   colptr::Vector{Ti}
#   colnnz::Vector{Ti}
#   rowval::Vector{Ti}
#   nzval::P
#   index_map::IndexMap{D}
#   function ParamTTInserterCSC(a::TTInserter{Algebra.InserterCSC{Tv,Ti},D},::Val{L}) where {Tv,Ti,D,L}
#     @unpack nrows,ncols,colptr,colnnz,rowval,nzval = a.inserter
#     index_map = a.index_map
#     pnzval = allocate_param_array(nzval,L)
#     P = typeof(pnzval)
#     new{Tv,Ti,P,D}(nrows,ncols,colptr,colnnz,rowval,pnzval,index_map)
#   end
# end

# Algebra.LoopStyle(::Type{<:ParamTTInserterCSC}) = Loop()

# @inline function Algebra.add_entry!(::typeof(+),a::ParamTTInserterCSC{Tv,Ti},v::Nothing,i,j)  where {Tv,Ti}
#   pini = Int(a.colptr[j])
#   pend = pini + Int(a.colnnz[j]) - 1
#   p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
#   if (p>pend)
#     # add new entry
#     a.colnnz[j] += 1
#     a.rowval[p] = i
#   elseif a.rowval[p] != i
#     # shift one forward from p to pend
#     @check  pend+1 < Int(a.colptr[j+1])
#     for k in pend:-1:p
#       o = k + 1
#       a.rowval[o] = a.rowval[k]
#     end
#     # add new entry
#     a.colnnz[j] += 1
#     a.rowval[p] = i
#   end
#   nothing
# end

# @noinline function Algebra.add_entry!(
#   ::typeof(+),a::ParamTTInserterCSC{Tv,Ti,P},v::AbstractParamContainer{Tv},i,j) where {Tv,Ti,P}
#   pini = Int(a.colptr[j])
#   pend = pini + Int(a.colnnz[j]) - 1
#   p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
#   if (p>pend)
#     # add new entry
#     a.colnnz[j] += 1
#     a.rowval[p] = i
#     @inbounds for l = 1:length(P)
#       a.nzval[l][p] = v[l]
#     end
#   elseif a.rowval[p] != i
#     # shift one forward from p to pend
#     @check  pend+1 < Int(a.colptr[j+1])
#     for k in pend:-1:p
#       o = k + 1
#       a.rowval[o] = a.rowval[k]
#       @inbounds for l = 1:length(P)
#         a.nzval[l][o] = a.nzval[l][k]
#       end
#     end
#     # add new entry
#     a.colnnz[j] += 1
#     a.rowval[p] = i
#     @inbounds for l = 1:length(P)
#       a.nzval[l][p] = v[l]
#     end
#   else
#     # update existing entry
#     @inbounds for l = 1:length(P)
#       a.nzval[l][p] += v[l]
#     end
#   end
#   nothing
# end

# function Algebra.create_from_nz(a::ParamTTInserterCSC{Tv,Ti,P,D}) where {Tv,Ti,P,D}
#   k = 1
#   for j in 1:a.ncols
#     pini = Int(a.colptr[j])
#     pend = pini + Int(a.colnnz[j]) - 1
#     for p in pini:pend
#       @inbounds for l = 1:length(P)
#         a.nzval[l][k] = a.nzval[l][p]
#       end
#       a.rowval[k] = a.rowval[p]
#       k += 1
#     end
#   end
#   @inbounds for j in 1:a.ncols
#     a.colptr[j+1] = a.colnnz[j]
#   end
#   length_to_ptrs!(a.colptr)
#   nnz = a.colptr[end]-1
#   resize!(a.rowval,nnz)
#   resize!(a.nzval,nnz)
#   csc = map(1:length(P)) do l
#     v = SparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,a.nzval[l])
#     TTArray(v,a.index_map)
#   end
#   ParamArray(csc)
# end
struct TTBuilder{A,D}
  builder::A
  index_map::IndexMap{D}
end

function Algebra.get_array_type(b::TTBuilder)
  T = get_array_type(b.builder)
  a = T(undef,tfill(0,Val(ndims(T)))...)
  return typeof(TTArray(a,b.index_map))
end

function Algebra.nz_counter(b::TTBuilder,axes)
  counter = nz_counter(b.builder,axes)
  TTCounter(counter,b.index_map)
end

struct TTSparseMatrixAssembler{A,D} <: SparseMatrixAssembler
  assem::A
  index_map::IndexMap{D}
end

FESpaces.num_rows(a::TTSparseMatrixAssembler) = num_rows(a.assem)
FESpaces.num_cols(a::TTSparseMatrixAssembler) = num_cols(a.assem)

function FESpaces.get_rows(a::TTSparseMatrixAssembler)
  return get_rows(a.assem)
end

function FESpaces.get_cols(a::TTSparseMatrixAssembler)
  return get_cols(a.assem)
end

function FESpaces.get_assembly_strategy(a::TTSparseMatrixAssembler)
  return get_assembly_strategy(a.assem)
end

function FESpaces.get_matrix_builder(a::TTSparseMatrixAssembler)
  builder = get_matrix_builder(a.assem)
  index_map = a.index_map
  return TTBuilder(builder,index_map)
end

function FESpaces.get_vector_builder(a::TTSparseMatrixAssembler)
  builder = get_vector_builder(a.assem)
  index_map = a.index_map
  return TTBuilder(builder,index_map)
end

function get_param_matrix_builder(
  a::TTSparseMatrixAssembler,
  r::AbstractParamRealization)

  mat = get_matrix_builder(a)
  M = get_array_type(mat)
  pmatrix_type = _get_param_matrix_type(M,r)
  SparseMatrixBuilder(pmatrix_type)
end

function get_param_vector_builder(
  a::TTSparseMatrixAssembler,
  r::AbstractParamRealization)

  vec = get_vector_builder(a)
  V = get_array_type(vec)
  pvector_type = _get_param_vector_type(V,r)
  ArrayBuilder(pvector_type)
end

function TTSparseMatrixAssembler(
  mat,
  vec,
  trial::T,
  test::TProductFESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()
  ) where T<:FESpace

  assem = GenericSparseMatrixAssembler(
    SparseMatrixBuilder(mat),
    ArrayBuilder(vec),
    get_free_dof_ids(test),
    get_free_dof_ids(trial),
    strategy)
  index_map = test.dof_permutation
  TTSparseMatrixAssembler(assem,index_map)
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::T,
  test::TProductFESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()
  ) where T<:FESpace # add some condition on T

  TTSparseMatrixAssembler(mat,vec,trial,test,strategy)
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

# function Algebra.allocate_coo_vectors(::Type{TTSparseMatrix{D,T,V}},n::Integer) where {D,T,V}
#   TTArray(Algebra.allocate_coo_vectors(V,n),Val(D))
# end

# function Algebra.sparse_from_coo(::Type{TTSparseMatrix{D,T,V}},i,j,v,m,n) where {D,T,V}
#   TTArray(Algebra.sparse_from_coo(V,i,j,v,m,n),Val(D))
# end

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
  index_map::IndexMap{D}
end

Algebra.LoopStyle(::Type{TTCounter{C,D}}) where {C,D} = LoopStyle(C)

@inline function Algebra.add_entry!(::typeof(+),a::TTCounter,v,i,j)
  add_entry!(+,a.counter,v,i,j)
end

function Algebra.nz_allocation(a::TTCounter{C,D}) where {C,D}
  inserter = nz_allocation(a.counter)
  TTInserter(inserter,a.index_map)
end

struct TTInserter{I,D}
  inserter::I
  index_map::IndexMap{D}
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
  TTArray(nz,a.index_map)
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
  index_map::IndexMap{D}
  function ParamTTInserterCSC(a::TTInserter{Algebra.InserterCSC{Tv,Ti},D},::Val{L}) where {Tv,Ti,D,L}
    @unpack nrows,ncols,colptr,colnnz,rowval,nzval = a.inserter
    index_map = a.index_map
    pnzval = allocate_param_array(nzval,L)
    P = typeof(pnzval)
    new{Tv,Ti,P,D}(nrows,ncols,colptr,colnnz,rowval,pnzval,index_map)
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
    TTArray(v,a.index_map)
  end
  ParamArray(csc)
end
