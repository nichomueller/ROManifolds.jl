"""
    reduced_fe_space(solver::RBSolver,feop::ParamFEOperator,s::AbstractSteadySnapshots
      ) -> (FESubspace, FESubspace)
    reduced_fe_space(solver::RBSolver,feop::TransientParamFEOperator,s::AbstractTransientSnapshots
      ) -> (FESubspace, FESubspace)
    reduced_fe_space(solver::RBSolver,feop::TransientParamFEOperator,s::BlockSnapshots
      ) -> (FESubspace, FESubspace)

Computes the subspace of the test, trial FE spaces contained in the FE operator
`feop` by compressing the snapshots `s`

"""
function reduced_fe_space(solver::RBSolver,feop,s)
  reduced_fe_space(get_state_reduction(solver),feop,s)
end

function reduced_fe_space(solver::AbstractReduction,feop,s)
  state_reduction = get_state_reduction(solver)
  timer = get_timer(solver)
  name = get_name(timer)
  reset_timer!(timer)

  soff = select_snapshots(s,num_snaps(state_reduction))
  @timeit timer name begin
    basis = reduced_basis(state_reduction,feop,soff,norm_matrix)
  end
  reduced_trial = fe_subspace(get_trial(feop),basis)
  reduced_test = fe_subspace(get_test(feop),basis)

  return reduced_trial,reduced_test
end

"""
    reduced_basis(red,s::AbstractSnapshots,args...) -> (Projection, Projection)

Computes the bases spanning the subspace of test, trial FE spaces by compressing
the snapshots `s`

"""
function reduced_basis(red::AbstractReduction,s::AbstractSnapshots,args...)
  Projection(red,s,args...)
end

function reduced_basis(red::AbstractReduction,feop,s::AbstractSnapshots)
  reduced_basis(s,norm_matrix)
end

function reduced_basis(red::NormedReduction,feop,s::AbstractSnapshots)
  norm_matrix = assemble_from_form(feop,get_norm(red))
  reduced_basis(s,norm_matrix)
end

function reduced_basis(red::SupremizerReduction,feop,s::AbstractSnapshots)
  supr_matrix = assemble_from_form(feop,get_supr(red))
  norm_matrix = assemble_from_form(feop,get_norm(red))
  basis = reduced_basis(s,norm_matrix)
  enrich_basis(basis,norm_matrix)
end

function fe_subspace(space::FESpace,basis)
  @abstractmethod
end

"""
    abstract type FESubspace <: FESpace end

Represents a vector subspace of a FE space.

Subtypes:
- [`RBSpace`](@ref)
- [`MultiFieldRBSpace`](@ref)
- [`BlockRBSpace`](@ref)

"""
abstract type FESubspace <: FESpace end

(U::FESubspace)(μ) = evaluate(U,μ)

get_space(r::FESubspace) = r.space
get_basis(r::FESubspace) = r.basis

get_basis_space(r::FESubspace) = get_basis_space(r.basis)
num_space_dofs(r::FESubspace) = num_space_dofs(r.basis)
num_reduced_space_dofs(r::FESubspace) = num_reduced_space_dofs(r.basis)

num_fe_free_dofs(r::FESubspace) = num_fe_dofs(r.basis)

FESpaces.num_free_dofs(r::FESubspace) = num_reduced_dofs(r.basis)

FESpaces.get_free_dof_ids(r::FESubspace) = Base.OneTo(num_free_dofs(r))

FESpaces.get_dirichlet_dof_ids(r::FESubspace) = get_dirichlet_dof_ids(r.space)

FESpaces.num_dirichlet_dofs(r::FESubspace) = num_dirichlet_dofs(r.space)

FESpaces.num_dirichlet_tags(r::FESubspace) = num_dirichlet_tags(r.space)

FESpaces.get_dirichlet_dof_tag(r::FESubspace) = get_dirichlet_dof_tag(r.space)

FESpaces.get_vector_type(r::FESubspace) = get_vector_type(r.space)

FESpaces.get_dof_value_type(r::FESubspace) = get_dof_value_type(r.space)

function Algebra.allocate_in_domain(r::FESubspace)
  zero_free_values(r)
end

function Algebra.allocate_in_range(r::FESubspace)
  zero_free_values(r.space)
end

function Arrays.evaluate(U::FESubspace,args...)
  space = evaluate(U.space,args...)
  fe_subspace(space,U.basis)
end

struct RecastMap <: Map end

function IndexMaps.recast(x::AbstractVector,r::FESubspace)
  cache = return_cache(RecastMap(),x,r)
  evaluate!(cache,RecastMap(),x,r)
  return cache
end

function Arrays.return_cache(k::RecastMap,x::AbstractParamVector,r::FESubspace)
  allocate_in_range(r)
end

"""
    RBSpace{A<:SingleFieldFESpace,B<:SteadyProjection} <: FESubspace

Reduced basis subspace in a steady setting

"""
struct RBSpace{A<:SingleFieldFESpace,B<:SteadyProjection} <: FESubspace
  space::A
  basis::B
end

function fe_subspace(space::SingleFieldFESpace,basis::Projection)
  RBSpace(space,basis)
end

function Arrays.evaluate!(cache,k::RecastMap,x::AbstractParamVector,r::RBSpace)
  @inbounds for ip in eachindex(x)
    cache[ip] = recast(x[ip],r.basis)
  end
end

# multi field interface

"""
    MultiFieldRBSpace{A<:MultiFieldFESpace,B<:BlockProjection} <: FESubspace

Reduced basis subspace in a MultiField setting

"""
struct MultiFieldRBSpace{A<:MultiFieldFESpace,B<:BlockProjection} <: FESubspace
  space::A
  basis::B
end

function fe_subspace(space::MultiFieldFESpace,basis::BlockProjection)
  MultiFieldRBSpace(space,basis)
end

function Base.getindex(r::MultiFieldRBSpace,i...)
  if isa(r.space,MultiFieldFESpace)
    fs = r.space
  else
    fs = evaluate(r.space,nothing)
  end
  return fe_subspace(fs.spaces[i...],r.basis[i...])
end

function Base.iterate(r::MultiFieldRBSpace)
  if isa(r.space,MultiFieldFESpace)
    fs = r.space
  else
    fs = evaluate(r.space,nothing)
  end
  i = 1
  ri = fe_subspace(fs.spaces[i],r.basis[i])
  state = i+1,fs
  return ri,state
end

function Base.iterate(r::MultiFieldRBSpace,state)
  i,fs = state
  if i > length(fs.spaces)
    return nothing
  end
  ri = fe_subspace(fs.spaces[i],r.basis[i])
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
  errors = [pod_error(r[i],s[i],norm_matrix[Block(i,i)]) for i in active_block_ids]
  return_cache(block_map,errors...)
end
