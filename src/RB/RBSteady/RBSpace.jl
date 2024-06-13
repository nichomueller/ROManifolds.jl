function reduced_fe_space(solver,feop,s)
  soff = select_snapshots(s,offline_params(solver))
  norm_matrix = assemble_norm_matrix(feop)
  basis = reduced_basis(feop,soff,norm_matrix;ϵ=get_tol(solver))
  reduced_trial = fe_subspace(get_trial(feop),basis)
  reduced_test = fe_subspace(get_test(feop),basis)
  return reduced_trial,reduced_test
end

function reduced_basis(s::AbstractSnapshots,args...;kwargs...)
  Projection(s,args...;kwargs...)
end

function reduced_basis(
  feop::ParamFEOperator,s::AbstractSnapshots,norm_matrix;kwargs...)
  reduced_basis(s,norm_matrix;kwargs...)
end

function reduced_basis(
  feop::ParamSaddlePointFEOp,s::AbstractSnapshots,norm_matrix;kwargs...)
  bases = reduced_basis(feop.op,s,norm_matrix;kwargs...)
  enrich_basis(feop,bases,norm_matrix)
end

function reduced_basis(
  feop::ParamFEOperatorWithTrian,s::AbstractSnapshots,norm_matrix;kwargs...)
  reduced_basis(feop.op,s,norm_matrix;kwargs...)
end

function reduced_basis(
  feop::ParamLinearNonlinearFEOperator,s::AbstractSnapshots,norm_matrix;kwargs...)
  reduced_basis(join_operators(feop),s,norm_matrix;kwargs...)
end

function enrich_basis(feop::ParamFEOperator,bases,norm_matrix)
  supr_op = assemble_coupling_matrix(feop)
  enrich_basis(bases,norm_matrix,supr_op)
end

function fe_subspace(space::FESpace,basis)
  RBSpace(space,basis)
end

abstract type FESubspace <: FESpace end

struct RBSpace{A,B} <: FESubspace
  space::A
  basis::B
end

function Arrays.evaluate(U::RBSpace,args...)
  space = evaluate(U.space,args...)
  RBSpace(space,U.basis)
end

(U::RBSpace)(μ) = evaluate(U,μ)

get_space(r::RBSpace) = r.space
get_basis(r::RBSpace) = r.basis

get_basis_space(r::RBSpace) = get_basis_space(r.basis)
num_space_dofs(r::RBSpace) = num_space_dofs(r.basis)
num_reduced_space_dofs(r::RBSpace) = num_reduced_space_dofs(r.basis)

num_fe_free_dofs(r::RBSpace) = num_fe_dofs(r.basis)

FESpaces.num_free_dofs(r::RBSpace) = num_reduced_dofs(r.basis)

FESpaces.get_free_dof_ids(r::RBSpace) = Base.OneTo(num_free_dofs(r))

FESpaces.get_dirichlet_dof_ids(r::RBSpace) = get_dirichlet_dof_ids(r.space)

FESpaces.num_dirichlet_dofs(r::RBSpace) = num_dirichlet_dofs(r.space)

FESpaces.num_dirichlet_tags(r::RBSpace) = num_dirichlet_tags(r.space)

FESpaces.get_dirichlet_dof_tag(r::RBSpace) = get_dirichlet_dof_tag(r.space)

FESpaces.get_vector_type(r::RBSpace) = get_vector_type(r.space)

function Algebra.allocate_in_domain(r::RBSpace)
  zero_free_values(r)
end

function Algebra.allocate_in_range(r::RBSpace)
  zero_free_values(r.space)
end

function ParamDataStructures.recast(x::AbstractVector,r::RBSpace)
  cache = return_cache(RecastMap(),x,r)
  evaluate!(cache,RecastMap(),x,r)
  return cache
end

struct RecastMap <: Map end

function Arrays.return_cache(k::RecastMap,x::AbstractParamVector,r::RBSpace)
  allocate_in_range(r)
end

function Arrays.evaluate!(cache,k::RecastMap,x::AbstractParamVector,r::RBSpace)
  @inbounds for ip in eachindex(x)
    cache[ip] .= recast(x[ip],r.basis)
  end
end

# multi field interface

const MultiFieldRBSpace{A,B<:BlockProjection} = RBSpace{A,B}

function Base.getindex(r::MultiFieldRBSpace,i...)
  if isa(r.space,MultiFieldFESpace)
    fs = r.space
  else
    fs = evaluate(r.space,nothing)
  end
  return RBSpace(fs.spaces[i...],r.basis[i...])
end

function Base.iterate(r::MultiFieldRBSpace)
  if isa(r.space,MultiFieldFESpace)
    fs = r.space
  else
    fs = evaluate(r.space,nothing)
  end
  i = 1
  ri = RBSpace(fs.spaces[i],r.basis[i])
  state = i+1,fs
  return ri,state
end

function Base.iterate(r::MultiFieldRBSpace,state)
  i,fs = state
  if i > length(fs.spaces)
    return nothing
  end
  ri = RBSpace(fs.spaces[i],r.basis[i])
  state = i+1,fs
  return ri,state
end

MultiField.MultiFieldStyle(r::MultiFieldRBSpace) = MultiFieldStyle(r.space)

function FESpaces.get_free_dof_ids(r::MultiFieldRBSpace)
  get_free_dof_ids(r,MultiFieldStyle(r))
end

function FESpaces.get_free_dof_ids(r::MultiFieldRBSpace,::ConsecutiveMultiFieldStyle)
  @notimplemented
end

function FESpaces.get_free_dof_ids(r::MultiFieldRBSpace,::BlockMultiFieldStyle{NB}) where NB
  block_num_dofs = map(range->num_free_dofs(r[range]),1:NB)
  return BlockArrays.blockedrange(block_num_dofs)
end

function get_touched_blocks(r::MultiFieldRBSpace)
  get_touched_blocks(r.basis)
end

function Arrays.return_cache(k::RecastMap,x::BlockVectorOfVectors,r::MultiFieldRBSpace)
  block_cache = [return_cache(k,x[Block(i)],r[i]) for i = 1:blocklength(x)]
  mortar(block_cache)
end

function Arrays.evaluate!(cache,k::RecastMap,x::BlockVectorOfVectors,r::MultiFieldRBSpace)
  @inbounds for i = 1:blocklength(cache)
    evaluate!(cache[Block(i)],k,x[Block(i)],r[i])
  end
  return cache
end

# for testing/visualization purposes

function pod_error(r::RBSpace,s::AbstractSnapshots,norm_matrix::AbstractMatrix)
  basis_space = get_basis_space(r)
  err_space = norm(s - basis_space*basis_space'*norm_matrix*s) / norm(s)
  Dict("err_space"=>err_space)
end

function pod_error(r::MultiFieldRBSpace,s::BlockSnapshots,norm_matrix::BlockMatrix)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  errors = Any[pod_error(r[i],s[i],norm_matrix[Block(i,i)]) for i in active_block_ids]
  return_cache(block_map,errors...)
end
