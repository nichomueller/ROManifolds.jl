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

function reduced_basis(s::AbstractSnapshots,args...;kwargs...)
  Projection(s,args...;kwargs...)
end

function reduced_basis(
  feop::TransientParamFEOperator,s::S,norm_matrix;kwargs...) where S
  reduced_basis(s,norm_matrix;kwargs...)
end

function reduced_basis(
  feop::TransientParamSaddlePointFEOperator,s::S,norm_matrix;kwargs...) where S
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

ODETools.∂t(U::RBSpace) = RBSpace(∂t(U),U.basis)
ODETools.∂tt(U::RBSpace) = RBSpace(∂tt(U),U.basis)

get_basis_space(r::RBSpace) = get_basis_space(r.basis)
num_space_dofs(r::RBSpace) = size(get_basis_space(r),1)
num_reduced_space_dofs(r::RBSpace) = size(get_basis_space(r),2)

get_basis_time(r::RBSpace) = get_basis_time(r.basis)
FEM.num_times(r::RBSpace) = size(get_basis_time(r),1)
num_reduced_times(r::RBSpace) = size(get_basis_time(r),2)

num_fe_free_dofs(r::RBSpace) = dot(num_space_dofs(r),num_times(r))

FESpaces.num_free_dofs(r::RBSpace) = dot(num_reduced_space_dofs(r),num_reduced_times(r))

FESpaces.get_free_dof_ids(r::RBSpace) = Base.OneTo(num_free_dofs(r))

FESpaces.get_dirichlet_dof_ids(r::RBSpace) = get_dirichlet_dof_ids(r.space)

FESpaces.num_dirichlet_dofs(r::RBSpace) = num_dirichlet_dofs(r.space)

FESpaces.num_dirichlet_tags(r::RBSpace) = num_dirichlet_tags(r.space)

FESpaces.get_dirichlet_dof_tag(r::RBSpace) = get_dirichlet_dof_tag(r.space)

function FESpaces.get_vector_type(r::RBSpace)
  change_length(x) = x
  change_length(::Type{ParamVector{T,A,L}}) where {T,A,L} = ParamVector{T,A,Int(L/num_times(r))}
  change_length(::Type{<:ParamBlockVector{T,A,L}}) where {T,A,L} = ParamBlockVector{T,A,Int(L/num_times(r))}
  V = get_vector_type(r.space)
  newV = change_length(V)
  return newV
end

function Algebra.allocate_in_domain(r::RBSpace)
  V = get_vector_type(r.space)
  allocate_vector(V,num_fe_free_dofs(r))
end

function compress_basis(b::Projection,test::RBSpace;kwargs...)
  compress_basis(b,test.basis;kwargs...)
end

function compress_basis(b::Projection,trial::RBSpace,test::RBSpace;kwargs...)
  compress_basis(b,trial.basis,test.basis;kwargs...)
end

struct RecastMap <: Map end

function Arrays.return_cache(k::RecastMap,x::ParamVector,r::RBSpace)
  allocate_in_domain(r)
end

function Arrays.evaluate!(cache,k::RecastMap,x::ParamVector,r::RBSpace)
  for ip in eachindex(x)
    Xip = recast(x[ip],r)
    for it in 1:num_times(r)
      cache[(ip-1)*num_times(r)+it] = Xip[:,it]
    end
  end
end

function recast(x::Vector,r::RBSpace)
  recast(x,r.basis)
end

function recast(x::AbstractVector,r::RBSpace)
  cache = return_cache(RecastMap(),x,r)
  evaluate!(cache,RecastMap(),x,r)
  return cache
end

# multi field interface

const BlockRBSpace = RBSpace{A,B} where {A,B<:BlockProjection}

function Base.getindex(r::BlockRBSpace,i...)
  if isa(space,MultiFieldFESpace)
    fs = r.space
  else
    fs = evaluate(r.space,nothing)
  end
  return RBSpace(fs.spaces[i...],r.basis[i...])
end

function Base.iterate(r::BlockRBSpace)
  if isa(space,MultiFieldFESpace)
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

function enrich_basis(feop::TransientParamFEOperator,bases,norm_matrix)
  supr_op = assemble_coupling_matrix(feop)
  enrich_basis(bases,norm_matrix,supr_op)
end

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
