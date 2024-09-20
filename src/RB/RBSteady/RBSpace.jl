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
  red = get_state_reduction(solver)
  soff = select_snapshots(s,offline_params(solver))
  reduced_fe_space(red,feop,soff)
end

function reduced_fe_space(red::AbstractReduction,feop,s)
  t = @timed begin
    basis = reduced_basis(red,feop,s)
  end
  reduced_trial = fe_subspace(get_trial(feop),basis)
  reduced_test = fe_subspace(get_test(feop),basis)

  println(CostTracker(t))

  return reduced_trial,reduced_test
end

"""
    reduced_basis(red,s::AbstractSnapshots,args...) -> (Projection, Projection)

Computes the bases spanning the subspace of test, trial FE spaces by compressing
the snapshots `s`

"""
function reduced_basis(
  red::AbstractReduction,
  s::AbstractArray,
  args...)

  projection(red,s,args...)
end

function reduced_basis(
  red::AbstractReduction,
  feop::GridapType,
  s::AbstractArray)

  reduced_basis(red,s)
end

function reduced_basis(
  red::AbstractReduction{<:ReductionStyle,EnergyNorm},
  feop::GridapType,
  s::AbstractArray)

  norm_matrix = assemble_matrix(feop,get_norm(red))
  reduced_basis(red,s,norm_matrix)
end

function reduced_basis(
  red::SupremizerReduction,
  feop::GridapType,
  s::AbstractArray)

  norm_matrix = assemble_matrix(feop,get_norm(red))
  supr_matrix = assemble_matrix(feop,get_supr(red))
  basis = reduced_basis(get_reduction(red),s,norm_matrix)
  enrich(red,basis,norm_matrix,supr_matrix)
end

function fe_subspace(space::FESpace,basis)
  @abstractmethod
end

"""
    abstract type FESubspace <: FESpace end

Represents a vector subspace of a FE space.

Subtypes:
- [`SingleFieldRBSpace`](@ref)
- [`MultiFieldRBSpace`](@ref)

"""
abstract type FESubspace <: FESpace end

(U::FESubspace)(μ) = evaluate(U,μ)

get_fe_space(r::FESubspace) = @abstractmethod
get_reduced_subspace(r::FESubspace) = @abstractmethod

get_basis(r::FESubspace) = get_basis(get_reduced_subspace(r))
num_fe_dofs(r::FESubspace) = num_fe_dofs(get_fe_space(r))

FESpaces.num_free_dofs(r::FESubspace) = num_reduced_dofs(get_reduced_subspace(r))

FESpaces.get_vector_type(r::FESubspace) = get_vector_type(get_fe_space(r))

function Algebra.allocate_in_domain(r::FESubspace)
  zero_free_values(r)
end

function Algebra.allocate_in_range(r::FESubspace)
  zero_free_values(get_fe_space(r))
end

function project(r::FESubspace,x::AbstractVector)
  project(get_reduced_subspace(r),x)
end

function project(r::FESubspace,x::AbstractParamVector)
  x̂ = allocate_in_domain(r)
  @inbounds for ip in eachindex(x)
    x̂[ip] = project(get_reduced_subspace(r),x[ip])
  end
  return x̂
end

function inv_project(r::FESubspace,x̂::AbstractVector)
  inv_project(get_reduced_subspace(r),x̂)
end

function inv_project(r::FESubspace,x̂::AbstractParamVector)
  x = allocate_in_range(r)
  @inbounds for ip in eachindex(x̂)
    x[ip] = inv_project(get_reduced_subspace(r),x̂[ip])
  end
  return x
end

function FESubspaceFunction(r::FESubspace,x::AbstractVector)
  x̂ = project(get_reduced_subspace(r),x)
  return FESubspaceFunction(x̂,r)
end

function FEFunction(r::FESubspace,x̂::AbstractVector)
  x = inv_project(r,x̂)
  fe = get_fe_space(r)
  xdir = get_dirichlet_values(fe)
  return FEFunction(fe,x,xdir)
end

abstract type FESubspaceFunction <: FEFunction end

struct SingleFieldFESubspaceFunction <: FESubspaceFunction
  reduced_free_values::AbstractVector
  reduced_space::FESubspace
end

"""
    SingleFieldRBSpace{A<:SingleFieldFESpace,B<:Projection} <: FESubspace

Reduced basis subspace in a steady setting

"""
struct SingleFieldRBSpace{A<:SingleFieldFESpace,B<:Projection} <: FESubspace
  space::A
  subspace::B
end

function fe_subspace(space::SingleFieldFESpace,basis::Projection)
  SingleFieldRBSpace(space,basis)
end

get_fe_space(r::SingleFieldRBSpace) = r.space
get_reduced_subspace(r::SingleFieldRBSpace) = r.subspace

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

# dealing with the transient case here

function Arrays.evaluate(r::FESubspace,args...)
  space = evaluate(get_fe_space(r),args...)
  subspace = fe_subspace(space,get_reduced_subspace(r))
  EvalFESubspace(subspace,args...)
end

struct EvalFESubspace{A<:FESubspace,B<:AbstractRealization} <: FESubspace
  subspace::A
  realization::B
end

get_fe_space(r::EvalFESubspace) = get_fe_space(r.subspace)
get_reduced_subspace(r::EvalFESubspace) = get_reduced_subspace(r.subspace)
