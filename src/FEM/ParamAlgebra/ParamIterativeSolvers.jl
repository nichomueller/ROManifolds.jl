# function Algebra.symbolic_setup(
#   solver::GridapSolvers.BlockTriangularSolver,
#   mat::ParamBlockMatrix)

#   mat_blocks   = blocks(mat)
#   block_caches = map(BlockSolvers.instantiate_block_cache,solver.blocks,mat_blocks)
#   block_ss     = map(symbolic_setup,solver.solvers,diag(block_caches))
#   return BlockSolvers.BlockTriangularSolverSS(solver,block_ss,block_caches)
# end

# function Algebra.symbolic_setup(
#   solver::BlockSolvers.BlockTriangularSolver{T,N},
#   mat::ParamBlockMatrix,
#   x::ParamBlockVector) where {T,N}

#   mat_blocks   = blocks(mat)
#   vec_blocks   = blocks(x)
#   block_caches = map(CartesianIndices(solver.blocks)) do I
#     BlockSolvers.instantiate_block_cache(solver.blocks[I],mat_blocks[I],vec_blocks[I[2]])
#   end
#   block_ss     = map(symbolic_setup,solver.solvers,diag(block_caches),vec_blocks)
#   return BlockTriangularSolverSS(solver,block_ss,block_caches)
# end

# function Algebra.numerical_setup(
#   ss::BlockSolvers.BlockTriangularSolverSS,
#   mat::ParamBlockMatrix)

#   solver      = ss.solver
#   block_ns    = map(numerical_setup,ss.block_ss,diag(ss.block_caches))

#   y = mortar(map(allocate_in_domain,diag(ss.block_caches))); fill!(y,0.0)
#   w = allocate_in_range(mat); fill!(w,0.0)
#   work_caches = w, y
#   return BlockSolvers.BlockTriangularSolverNS(solver,block_ns,ss.block_caches,work_caches)
# end

# function Algebra.numerical_setup(
#   ss::BlockSolvers.BlockTriangularSolverSS,
#   mat::ParamBlockMatrix,
#   x::ParamBlockVector)

#   solver      = ss.solver
#   block_ns    = map(numerical_setup,ss.block_ss,diag(ss.block_caches),blocks(x))

#   y = mortar(map(allocate_in_domain,diag(ss.block_caches))); fill!(y,0.0)
#   w = allocate_in_range(mat); fill!(w,0.0)
#   work_caches = w, y
#   return BlockSolvers.BlockTriangularSolverNS(solver,block_ns,ss.block_caches,work_caches)
# end

# function Algebra.numerical_setup!(
#   ns::BlockSolvers.BlockTriangularSolverNS,
#   mat::ParamBlockMatrix)

#   solver       = ns.solver
#   mat_blocks   = blocks(mat)
#   block_caches = map(BlockSolvers.update_block_cache!,ns.block_caches,solver.blocks,mat_blocks)
#   map(diag(solver.blocks),ns.block_ns,diag(block_caches)) do bi, nsi, ci
#     if BlockSolvers.is_nonlinear(bi)
#       numerical_setup!(nsi,ci)
#     end
#   end
#   return ns
# end

# function Algebra.numerical_setup!(
#   ns::BlockSolvers.BlockTriangularSolverNS,
#   mat::ParamBlockMatrix,
#   x::ParamBlockVector)

#   solver       = ns.solver
#   mat_blocks   = blocks(mat)
#   vec_blocks   = blocks(x)
#   block_caches = map(CartesianIndices(solver.blocks)) do I
#     update_block_cache!(ns.block_caches[I],solver.blocks[I],mat_blocks[I],vec_blocks[I[2]])
#   end
#   map(diag(solver.blocks),ns.block_ns,diag(block_caches),vec_blocks) do bi, nsi, ci, xi
#     if BlockSolvers.is_nonlinear(bi)
#       numerical_setup!(nsi,ci,xi)
#     end
#   end
#   return ns
# end

function BlockSolvers.instantiate_block_cache(
  block::BlockSolvers.BiformBlock,
  mat::ParamMatrix)

  cache = assemble_matrix(block.f,block.assem,block.trial,block.test)
  return array_of_similar_arrays(cache,param_length(mat))
end

function BlockSolvers.instantiate_block_cache(
  block::TriformBlock,
  mat::ParamMatrix,
  x::ParamVector)

  @check param_length(mat) == param_length(vec)
  uh = FEFunction(block.trial,x)
  f(u,v) = block.f(uh,u,v)
  cache = assemble_matrix(f,block.assem,block.trial,block.test)
  return array_of_similar_arrays(cache,param_length(mat))
end

# function Algebra.solve!(
#   x::ParamBlockVector,
#   ns::BlockSolvers.BlockTriangularSolverNS{Val{:lower}},
#   b::ParamBlockVector)

#   @check blocklength(x) == blocklength(b) == length(ns.block_ns)
#   NB = length(ns.block_ns)
#   c = ns.solver.coeffs
#   w, y = ns.work_caches
#   mats = ns.block_caches
#   for iB in 1:NB
#     # Add lower off-diagonal contributions
#     wi  = w[Block(iB)]
#     copy!(wi,b[Block(iB)])
#     for jB in 1:iB-1
#       cij = c[iB,jB]
#       if abs(cij) > eps(cij)
#         xj = x[Block(jB)]
#         mul!(wi,mats[iB,jB],xj,-cij,1.0)
#       end
#     end

#     # Solve diagonal block
#     nsi = ns.block_ns[iB]
#     xi  = x[Block(iB)]
#     yi  = y[Block(iB)]
#     solve!(yi,nsi,wi)
#     copy!(xi,yi)
#   end
#   return x
# end

# function Algebra.solve!(
#   x::ParamBlockVector,
#   ns::BlockSolvers.BlockTriangularSolverNS{Val{:upper}},
#   b::ParamBlockVector)

#   @check blocklength(x) == blocklength(b) == length(ns.block_ns)
#   NB = length(ns.block_ns)
#   c = ns.solver.coeffs
#   w, y = ns.work_caches
#   mats = ns.block_caches
#   for iB in NB:-1:1
#     # Add upper off-diagonal contributions
#     wi  = w[Block(iB)]
#     copy!(wi,b[Block(iB)])
#     for jB in iB+1:NB
#       cij = c[iB,jB]
#       if abs(cij) > eps(cij)
#         xj = x[Block(jB)]
#         mul!(wi,mats[iB,jB],xj,-cij,1.0)
#       end
#     end

#     # Solve diagonal block
#     nsi = ns.block_ns[iB]
#     xi  = x[Block(iB)]
#     yi  = y[Block(iB)]
#     solve!(yi,nsi,wi)
#     copy!(xi,yi) # Remove this with PA 0.4
#   end
#   return x
# end
