"""
    reduced_spaces(solver::RBSolver,feop::ParamFEOperator,s::AbstractSnapshots
      ) -> (RBSpace, RBSpace)

Computes the subspace of the test, trial FE spaces contained in the FE operator
`feop` by compressing the snapshots `s`
"""
function reduced_spaces(solver::RBSolver,feop::ParamFEOperator,s::AbstractSnapshots)
  red = get_state_reduction(solver)
  soff = select_snapshots(s,offline_params(solver))
  reduced_spaces(red,feop,soff)
end

function reduced_spaces(red::Reduction,feop::ParamFEOperator,s::AbstractSnapshots)
  t = @timed begin
    basis = reduced_basis(red,feop,s)
  end
  println(CostTracker(t,name="Basis construction"))

  reduced_trial = reduced_subspace(get_trial(feop),basis)
  reduced_test = reduced_subspace(get_test(feop),basis)
  return reduced_trial,reduced_test
end

"""
    reduced_basis(red::Reduction,s::AbstractSnapshots,args...) -> (Projection, Projection)

Computes the bases spanning the subspace of test, trial FE spaces by compressing
the snapshots `s`
"""
function reduced_basis(
  red::Reduction,
  s::AbstractSnapshots,
  args...)

  projection(red,s,args...)
end

function reduced_basis(
  red::Reduction,
  feop::ParamFEOperator,
  s::AbstractSnapshots)

  reduced_basis(red,s)
end

function reduced_basis(
  red::Reduction{<:ReductionStyle,EnergyNorm},
  feop::ParamFEOperator,
  s::AbstractSnapshots)

  norm_matrix = assemble_matrix(feop,get_norm(red))
  reduced_basis(red,s,norm_matrix)
end

function reduced_basis(
  red::SupremizerReduction,
  feop::ParamFEOperator,
  s::AbstractSnapshots)

  norm_matrix = assemble_matrix(feop,get_norm(red))
  supr_matrix = assemble_matrix(feop,get_supr(red))
  basis = reduced_basis(get_reduction(red),s,norm_matrix)
  enrich!(red,basis,norm_matrix,supr_matrix)
  return basis
end

function reduced_subspace(space::FESpace,basis)
  @abstractmethod
end

"""
    abstract type RBSpace <: FESpace end

Represents a vector subspace of a FE space.

Subtypes:

- `SingleFieldRBSpace`
- `MultiFieldRBSpace`
"""
abstract type RBSpace <: FESpace end

(U::RBSpace)(μ) = evaluate(U,μ)

FESpaces.get_fe_space(r::RBSpace) = @abstractmethod
get_reduced_subspace(r::RBSpace) = @abstractmethod

function Arrays.evaluate(r::RBSpace,args...)
  space = evaluate(get_fe_space(r),args...)
  reduced_subspace(space,get_reduced_subspace(r))
end

get_basis(r::RBSpace) = get_basis(get_reduced_subspace(r))
num_fe_dofs(r::RBSpace) = num_fe_dofs(get_fe_space(r))

FESpaces.num_free_dofs(r::RBSpace) = num_reduced_dofs(get_reduced_subspace(r))

FESpaces.get_free_dof_ids(r::RBSpace) = Base.OneTo(num_free_dofs(r))

FESpaces.get_vector_type(r::RBSpace) = get_vector_type(get_fe_space(r))

for (f,f!,g) in zip(
  (:project,:inv_project),
  (:project!,:inv_project!),
  (:(Algebra.allocate_in_domain),:(Algebra.allocate_in_range)))

  @eval begin
    function $f(r::RBSpace,x::AbstractVector)
      y = $g(r,x)
      $f!(y,r,x)
      return y
    end

    function $f!(y,r::RBSpace,x::AbstractVector)
      $f!(y,get_reduced_subspace(r),x)
    end

    function $g(r::RBSpace,x::AbstractVector)
      $g(get_reduced_subspace(r),x)
    end
  end
end

function project(r::RBSpace,x::Projection)
  galerkin_projection(get_reduced_subspace(r),x)
end

function project(r1::RBSpace,x::Projection,r2::RBSpace)
  galerkin_projection(get_reduced_subspace(r1),x,get_reduced_subspace(r2))
end

get_norm_matrix(r::RBSpace) = get_norm_matrix(get_reduced_subspace(r))

function FESpaces.FEFunction(r::RBSpace,x̂::AbstractVector)
  x = inv_project(r,x̂)
  fe = get_fe_space(r)
  xdir = get_dirichlet_values(fe)
  return FEFunction(fe,x,xdir)
end

"""
    struct SingleFieldRBSpace{S,P} <: RBSpace
      space::S
      subspace::P
    end

Reduced basis subspace of a `SingleFieldFESpace` in `Gridap`
"""
struct SingleFieldRBSpace{S<:SingleFieldFESpace,P} <: RBSpace
  space::S
  subspace::P
end

function reduced_subspace(space::SingleFieldFESpace,basis::Projection)
  SingleFieldRBSpace(space,basis)
end

FESpaces.get_fe_space(r::SingleFieldRBSpace) = r.space
get_reduced_subspace(r::SingleFieldRBSpace) = r.subspace

const SingleFieldParamRBSpace{P} = SingleFieldRBSpace{<:SingleFieldParamFESpace,P}

ParamDataStructures.param_length(r::SingleFieldParamRBSpace) = param_length(r.space)

function FESpaces.zero_free_values(r::SingleFieldParamRBSpace)
  PV = get_vector_type(r)
  V = eltype(PV)
  fv = allocate_vector(V,get_free_dof_ids(r))
  L = param_length(r)
  consecutive_param_array(fv,L)
end

"""
    struct MultiFieldRBSpace <: RBSpace
      space::MultiFieldFESpace
      subspace::BlockProjection
    end

Reduced basis subspace of a `MultiFieldFESpace` in `Gridap`
"""
struct MultiFieldRBSpace <: RBSpace
  space::MultiFieldFESpace
  subspace::BlockProjection
end

function reduced_subspace(space::MultiFieldFESpace,subspace::BlockProjection)
  MultiFieldRBSpace(space,subspace)
end

FESpaces.get_fe_space(r::MultiFieldRBSpace) = r.space
get_reduced_subspace(r::MultiFieldRBSpace) = r.subspace

function Base.getindex(r::MultiFieldRBSpace,i::Integer)
  return reduced_subspace(r.space.spaces[i],r.subspace[i])
end

function Base.iterate(r::MultiFieldRBSpace,state=1)
  if state > num_fields(r)
    return nothing
  end
  ri = reduced_subspace(r.space[state],r.subspace[state])
  return ri,state+1
end

MultiField.MultiFieldStyle(r::MultiFieldRBSpace) = MultiFieldStyle(get_fe_space(r))
MultiField.num_fields(r::MultiFieldRBSpace) = num_fields(get_fe_space(r))
Base.length(r::MultiFieldRBSpace) = num_fields(r)

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

function FESpaces.zero_free_values(r::MultiFieldRBSpace)
  mortar(map(zero_free_values,r))
end

function FESpaces.zero_dirichlet_values(r::MultiFieldRBSpace)
  mortar(map(zero_dirichlet_values,r))
end

# utils

function to_snapshots(f::FESpace,x::AbstractParamVector,r::AbstractRealization)
  i = get_dof_map(f)
  Snapshots(x,i,r)
end

function to_snapshots(f::RBSpace,x̂::AbstractParamVector,r::AbstractRealization)
  fr = f(r)
  x = inv_project(fr,x̂)
  to_snapshots(get_fe_space(f),x,r)
end
