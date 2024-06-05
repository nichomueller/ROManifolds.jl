function BlockSolvers.instantiate_block_cache(
  block::BlockSolvers.BiformBlock,
  mat::AbstractParamMatrix)

  cache = assemble_matrix(block.f,block.assem,block.trial,block.test)
  return array_of_similar_arrays(cache,param_length(mat))
end

function BlockSolvers.instantiate_block_cache(
  block::TriformBlock,
  mat::AbstractParamMatrix,
  x::AbstractParamVector)

  @check param_length(mat) == param_length(vec)
  uh = FEFunction(block.trial,x)
  f(u,v) = block.f(uh,u,v)
  cache = assemble_matrix(f,block.assem,block.trial,block.test)
  return array_of_similar_arrays(cache,param_length(mat))
end
