function reduced_fe_space(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::S) where S

  soff = select_snapshots(s,offline_params(solver))
  norm_matrix = assemble_norm_matrix(feop)
  basis = reduced_basis(feop,soff,norm_matrix;ϵ=get_tol(solver))
  reduced_trial = fe_subspace(get_trial(feop),basis)
  reduced_test = fe_subspace(get_test(feop),basis)
  return reduced_trial,reduced_test
end

function reduced_basis(s::S,args...;kwargs...) where S
  Projection(s,args...;kwargs...)
end

function reduced_basis(
  feop::TransientParamFEOperator,s::S,norm_matrix;kwargs...) where S
  reduced_basis(s,norm_matrix;kwargs...)
end

function reduced_basis(
  feop::TransientParamSaddlePointFEOp,s::S,norm_matrix;kwargs...) where S
  bases = reduced_basis(feop.op,s,norm_matrix;kwargs...)
  enrich_basis(feop,bases,norm_matrix)
end

function reduced_basis(
  feop::TransientParamFEOperatorWithTrian,s::S,norm_matrix;kwargs...) where S
  reduced_basis(feop.op,s,norm_matrix;kwargs...)
end

function reduced_basis(
  feop::TransientParamLinearNonlinearFEOperator,s::S,norm_matrix;kwargs...) where S
  reduced_basis(join_operators(feop),s,norm_matrix;kwargs...)
end

function enrich_basis(feop::TransientParamFEOperator,bases,norm_matrix)
  supr_op = assemble_coupling_matrix(feop)
  enrich_basis(bases,norm_matrix,supr_op)
end

function fe_subspace(space,basis)
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

(U::RBSpace)(r) = evaluate(U,r)
(U::RBSpace)(μ,t) = evaluate(U,μ,t)

ODEs.time_derivative(U::RBSpace) = RBSpace(time_derivative(U),U.basis)

get_space(r::RBSpace) = r.space
get_basis(r::RBSpace) = r.basis

get_basis_space(r::RBSpace) = get_basis_space(r.basis)
num_space_dofs(r::RBSpace) = num_space_dofs(r.basis)
num_reduced_space_dofs(r::RBSpace) = num_reduced_space_dofs(r.basis)

get_basis_time(r::RBSpace) = get_basis_time(r.basis)
ParamDataStructures.num_times(r::RBSpace) = num_times(r.basis)
num_reduced_times(r::RBSpace) = num_reduced_times(r.basis)

num_fe_free_dofs(r::RBSpace) = num_fe_dofs(r.basis)

FESpaces.num_free_dofs(r::RBSpace) = num_reduced_dofs(r.basis)

FESpaces.get_free_dof_ids(r::RBSpace) = Base.OneTo(num_free_dofs(r))

FESpaces.get_dirichlet_dof_ids(r::RBSpace) = get_dirichlet_dof_ids(r.space)

FESpaces.num_dirichlet_dofs(r::RBSpace) = num_dirichlet_dofs(r.space)

FESpaces.num_dirichlet_tags(r::RBSpace) = num_dirichlet_tags(r.space)

FESpaces.get_dirichlet_dof_tag(r::RBSpace) = get_dirichlet_dof_tag(r.space)

function FESpaces.get_vector_type(r::RBSpace)
  change_length(x) = x
  change_length(::Type{ParamVector{T,L,A}}) where {T,L,A} = ParamVector{T,Int(L/num_times(r)),A}
  change_length(::Type{<:ParamBlockVector{T,L,A}}) where {T,L,A} = ParamBlockVector{T,Int(L/num_times(r)),A}
  V = get_vector_type(r.space)
  newV = change_length(V)
  return newV
end

function Algebra.allocate_in_domain(r::RBSpace)
  zero_free_values(r.space)
end

function recast(x::AbstractVector,r::RBSpace)
  cache = return_cache(RecastMap(),x,r)
  evaluate!(cache,RecastMap(),x,r)
  return cache
end

struct RecastMap <: Map end

function Arrays.return_cache(k::RecastMap,x::ParamVector,r::RBSpace)
  allocate_in_domain(r)
end

function Arrays.evaluate!(cache,k::RecastMap,x::ParamVector,r::RBSpace)
  for ip in eachindex(x)
    Xip = recast(x[ip],r.basis)
    for it in 1:num_times(r)
      cache[(it-1)*length(x)+ip] .= Xip[:,it]
    end
  end
end

function Arrays.evaluate!(cache::ParamTTArray,k::RecastMap,x::ParamVector,r::RBSpace)
  c = get_values(cache)
  evaluate!(c,k,x,r)
  copyto!(cache,c)
end

# multi field interface

const BlockRBSpace = RBSpace{A,B} where {A,B<:BlockProjection}

function Base.getindex(r::BlockRBSpace,i...)
  if isa(r.space,MultiFieldFESpace)
    fs = r.space
  else
    fs = evaluate(r.space,nothing)
  end
  return RBSpace(fs.spaces[i...],r.basis[i...])
end

function Base.iterate(r::BlockRBSpace)
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

function Base.iterate(r::BlockRBSpace,state)
  i,fs = state
  if i > length(fs.spaces)
    return nothing
  end
  ri = RBSpace(fs.spaces[i],r.basis[i])
  state = i+1,fs
  return ri,state
end

MultiField.MultiFieldStyle(r::BlockRBSpace) = MultiFieldStyle(r.space)

function FESpaces.get_free_dof_ids(r::BlockRBSpace)
  get_free_dof_ids(r,MultiFieldStyle(r))
end

function FESpaces.get_free_dof_ids(r::BlockRBSpace,::ConsecutiveMultiFieldStyle)
  @notimplemented
end

function FESpaces.get_free_dof_ids(r::BlockRBSpace,::BlockMultiFieldStyle{NB}) where NB
  block_num_dofs = map(range->num_free_dofs(r[range]),1:NB)
  return BlockArrays.blockedrange(block_num_dofs)
end

function FESpaces.zero_free_values(
  r::BlockRBSpace{<:MultiFieldParamFESpace{<:BlockMultiFieldStyle{NB}}}) where NB
  block_num_dofs = map(range->num_free_dofs(r[range]),1:NB)
  block_vtypes = map(range->get_vector_type(r[range]),1:NB)
  values = mortar(map(allocate_vector,block_vtypes,block_num_dofs))
  fill!(values,zero(eltype(values)))
  return values
end

function get_touched_blocks(r::BlockRBSpace)
  get_touched_blocks(r.basis)
end

num_fields(r::BlockRBSpace) = length(get_touched_blocks(r))

function Arrays.return_cache(k::RecastMap,x::ParamBlockVector,r::BlockRBSpace)
  block_cache = map(1:blocklength(x)) do i
    return_cache(k,x[Block(i)],r[i])
  end
  mortar(block_cache)
end

function Arrays.evaluate!(cache,k::RecastMap,x::ParamBlockVector,r::BlockRBSpace)
  @inbounds for i = 1:blocklength(cache)
    evaluate!(cache[Block(i)],k,x[Block(i)],r[i])
  end
  return cache
end

# for testing/visualization purposes

function pod_error(r::RBSpace,s::AbstractSnapshots,norm_matrix::AbstractMatrix)
  s2 = change_mode(s)
  basis_space = get_basis_space(r)
  basis_time = get_basis_time(r)
  err_space = norm(s - basis_space*basis_space'*norm_matrix*s) / norm(s)
  err_time = norm(s2 - basis_time*basis_time'*s2) / norm(s2)
  Dict("err_space"=>err_space,"err_time"=>err_time)
end

function pod_error(r::BlockRBSpace,s::BlockSnapshots,norm_matrix::BlockMatrix)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  errors = Any[pod_error(r[i],s[i],norm_matrix[Block(i,i)]) for i in active_block_ids]
  return_cache(block_map,errors...)
end
