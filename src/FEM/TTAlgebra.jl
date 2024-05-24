struct TTBuilder{A,B}
  builder::A
  index_map::B
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

struct TTSparseMatrixAssembler{A,B,C} <: SparseMatrixAssembler
  assem::A
  matrix_index_map::B
  vector_index_map::C
end

get_matrix_index_map(a::TTSparseMatrixAssembler) = a.matrix_index_map
get_vector_index_map(a::TTSparseMatrixAssembler) = a.vector_index_map

FESpaces.get_rows(a::TTSparseMatrixAssembler) = FESpaces.get_rows(a.assem)
FESpaces.get_cols(a::TTSparseMatrixAssembler) = FESpaces.get_cols(a.assem)

function FESpaces.get_assembly_strategy(a::TTSparseMatrixAssembler)
  return FESpaces.get_assembly_strategy(a.assem)
end

function FESpaces.get_matrix_builder(a::TTSparseMatrixAssembler)
  builder = get_matrix_builder(a.assem)
  index_map = get_matrix_index_map(a)
  return TTBuilder(builder,index_map)
end

function FESpaces.get_vector_builder(a::TTSparseMatrixAssembler)
  builder = get_vector_builder(a.assem)
  index_map = get_vector_index_map(a)
  return TTBuilder(builder,index_map)
end

function get_param_matrix_builder(
  a::TTSparseMatrixAssembler,
  r::AbstractParamRealization)

  builder = get_param_matrix_builder(a.assem,r)
  index_map = get_matrix_index_map(a)
  return TTBuilder(builder,index_map)
end

function get_param_vector_builder(
  a::TTSparseMatrixAssembler,
  r::AbstractParamRealization)

  builder = get_param_vector_builder(a.assem,r)
  index_map = get_vector_index_map(a)
  return TTBuilder(builder,index_map)
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
  matrix_index_map = get_sparse_index_map(trial,test)
  vector_index_map = get_dof_permutation(test)
  TTSparseMatrixAssembler(assem,matrix_index_map,vector_index_map)
end

for F in (:TrialFESpace,:TransientTrialFESpace,:TrialParamFESpace,:FESpaceToParamFESpace,:TransientTrialParamFESpace)
  @eval begin
    function FESpaces.SparseMatrixAssembler(
      mat,
      vec,
      trial::$F{<:TProductFESpace},
      test::TProductFESpace,
      strategy::AssemblyStrategy=DefaultAssemblyStrategy()
      )

      TTSparseMatrixAssembler(mat,vec,trial,test,strategy)
    end
  end
end

@inline function Algebra.add_entry!(combine::Function,A::TTArray,v::Number,i)
  add_entry!(combine,A.values,v,i)
  A
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

struct TTCounter{A,B}
  counter::A
  index_map::B
end

Algebra.LoopStyle(::Type{TTCounter{A,B}}) where {A,B} = LoopStyle(A)

@inline function Algebra.add_entry!(::typeof(+),a::TTCounter,v,i)
  add_entry!(+,a.counter,v,i)
end

@inline function Algebra.add_entry!(::typeof(+),a::TTCounter,v,i,j)
  add_entry!(+,a.counter,v,i,j)
end

function Algebra.nz_allocation(a::TTCounter{<:Algebra.ArrayCounter})
  TTArray(nz_allocation(a.counter),get_vector_index_map(a))
end

function Algebra.nz_allocation(a::TTCounter)
  inserter = nz_allocation(a.counter)
  TTInserter(inserter,get_matrix_index_map(a))
end

struct TTInserter{A,B}
  inserter::A
  index_map::B
end

Algebra.LoopStyle(::Type{TTInserter{A,B}}) where {A,B} = LoopStyle(A)

@inline function Algebra.add_entry!(::typeof(+),a::TTInserter,v::Nothing,i)
  Algebra.add_entry!(+,a.inserter,v,i)
end

@noinline function Algebra.add_entry!(::typeof(+),a::TTInserter,v,i)
  Algebra.add_entry!(+,a.inserter,v,i)
end

@inline function Algebra.add_entry!(::typeof(+),a::TTInserter,v::Nothing,i,j)
  Algebra.add_entry!(+,a.inserter,v,i,j)
end

@noinline function Algebra.add_entry!(::typeof(+),a::TTInserter,v,i,j)
  Algebra.add_entry!(+,a.inserter,v,i,j)
end

function Algebra.create_from_nz(a::TTInserter)
  nz = create_from_nz(a.inserter)
  TTArray(nz,a.index_map)
end

# Param - TT interface

function Algebra.nz_allocation(a::TTCounter{<:Algebra.ArrayCounter{<:ParamVector{T,L,A}}}) where {T,L,A}
  counter = a.counter
  index_map = a.index_map
  elA = eltype(A)
  v = fill!(similar(elA,map(length,counter.axes)),zero(T))
  ttv = TTArray(v,index_map)
  allocate_param_array(ttv,L)
end

function Algebra.nz_allocation(a::TTCounter{<:ParamCounter})
  counter = nz_allocation(a.counter)
  index_map = a.index_map
  ParamTTInserterCSC(counter,index_map)
end

struct ParamTTInserterCSC{A,B}
  inserter::A
  index_map::B
end

Algebra.LoopStyle(::Type{ParamTTInserterCSC{A,B}}) where {A,B} = LoopStyle(A)

@inline function Algebra.add_entry!(::typeof(+),a::ParamTTInserterCSC,v::Nothing,i,j)
  add_entry!(+,a.inserter,v,i,j)
end

@noinline function Algebra.add_entry!(::typeof(+),a::ParamTTInserterCSC,v::AbstractParamContainer,i,j)
  add_entry!(+,a.inserter,v,i,j)
end

function Algebra.create_from_nz(a::ParamTTInserterCSC)
  inserter = a.inserter
  index_map = a.index_map
  k = 1
  for j in 1:inserter.ncols
    pini = Int(inserter.colptr[j])
    pend = pini + Int(inserter.colnnz[j]) - 1
    for p in pini:pend
      @inbounds for l = eachindex(inserter.nzval)
        inserter.nzval[l][k] = inserter.nzval[l][p]
      end
      inserter.rowval[k] = inserter.rowval[p]
      k += 1
    end
  end
  @inbounds for j in 1:inserter.ncols
    inserter.colptr[j+1] = inserter.colnnz[j]
  end
  length_to_ptrs!(inserter.colptr)
  nnz = inserter.colptr[end]-1
  resize!(inserter.rowval,nnz)
  resize!(inserter.nzval,nnz)
  csc = map(eachindex(inserter.nzval)) do l
    v = SparseMatrixCSC(inserter.nrows,inserter.ncols,inserter.colptr,inserter.rowval,inserter.nzval[l])
    TTArray(v,index_map)
  end
  ParamArray(csc)
end
