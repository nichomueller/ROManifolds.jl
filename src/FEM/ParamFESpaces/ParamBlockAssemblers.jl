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
