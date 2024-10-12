function get_param_matrix_builder(
  a::MultiField.BlockSparseMatrixAssembler,
  r::AbstractParamRealization)

  assem = first(a.block_assemblers)
  get_param_matrix_builder(assem,r)
end

function get_param_vector_builder(
  a::MultiField.BlockSparseMatrixAssembler,
  r::AbstractParamRealization)

  assem = first(a.block_assemblers)
  get_param_vector_builder(assem,r)
end

function get_param_assembler(
  a::MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P},
  r::AbstractParamRealization) where {NB,NV,SB,P}

  matrix_builder = get_param_matrix_builder(a,r)
  vector_builder = get_param_vector_builder(a,r)
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  block_idx = CartesianIndices((NB,NB))
  block_assemblers = map(block_idx) do idx
    r = rows[idx[1]]
    c = cols[idx[2]]
    s = strategy[idx[1],idx[2]]
    GenericSparseMatrixAssembler(matrix_builder,vector_builder,r,c,s)
  end
  MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P}(block_assemblers)
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
