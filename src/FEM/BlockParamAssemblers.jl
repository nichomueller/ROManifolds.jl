function Algebra.allocate_vector(
  ::Type{<:ParamBlockVector{T,V}},
  indices::BlockedUnitRange) where {T,V}

  mortar(map(ids -> allocate_vector(V,ids),blocks(indices)))
end

function Algebra.allocate_in_range(matrix::ParamBlockMatrix{T,A,L}) where {T,A,L}
  BV = BlockVector{T,Vector{ParamVector{T,Vector{eltype(A)},L}}}
  V = ParamBlockVector{T,Vector{eltype(A)},L,BV}
  allocate_in_range(V,matrix)
end

function Algebra.allocate_in_domain(matrix::ParamBlockMatrix{T,A,L}) where {T,A,L}
  BV = BlockVector{T,Vector{ParamVector{T,Vector{eltype(A)},L}}}
  V = ParamBlockVector{T,Vector{eltype(A)},L,BV}
  allocate_in_domain(V,matrix)
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::MultiFieldParamFESpace{MS},
  test::MultiFieldFESpace{MS},
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()
  ) where MS <: BlockMultiFieldStyle

  N = length_free_values(trial)
  pmat = typeof(ParamMatrix{mat}(undef,N))
  pvec = typeof(ParamVector{vec}(undef,N))
  mfs = MultiFieldStyle(test)
  MultiField.BlockSparseMatrixAssembler(
    mfs,
    trial,
    test,
    SparseMatrixBuilder(pmat),
    ArrayBuilder(pvec),
    strategy)
end

function FESpaces.assemble_vector_add!(
  b::ParamBlockVector,
  a::MultiField.BlockSparseMatrixAssembler,
  vecdata)
  b1 = ArrayBlock(blocks(b),fill(true,blocksize(b)))
  b2 = MultiField.expand_blocks(a,b1)
  FESpaces.assemble_vector_add!(b2,a,vecdata)
end

function FESpaces.assemble_matrix_add!(
  mat::ParamBlockMatrix,
  a::MultiField.BlockSparseMatrixAssembler,
  matdata)
  m1 = ArrayBlock(blocks(mat),fill(true,blocksize(mat)))
  m2 = MultiField.expand_blocks(a,m1)
  FESpaces.assemble_matrix_add!(m2,a,matdata)
end

function FESpaces.assemble_matrix_and_vector_add!(
  A::ParamBlockMatrix,
  b::ParamBlockVector,
  a::MultiField.BlockSparseMatrixAssembler,
  data)
  m1 = ArrayBlock(blocks(A),fill(true,blocksize(A)))
  m2 = MultiField.expand_blocks(a,m1)
  b1 = ArrayBlock(blocks(b),fill(true,blocksize(b)))
  b2 = MultiField.expand_blocks(a,b1)
  FESpaces.assemble_matrix_and_vector_add!(m2,b2,a,data)
end
