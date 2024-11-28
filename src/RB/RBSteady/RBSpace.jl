"""
    reduced_fe_space(solver::RBSolver,feop::ParamFEOperator,s::AbstractSteadySnapshots
      ) -> (RBSpace, RBSpace)
    reduced_fe_space(solver::RBSolver,feop::TransientParamFEOperator,s::AbstractTransientSnapshots
      ) -> (RBSpace, RBSpace)
    reduced_fe_space(solver::RBSolver,feop::TransientParamFEOperator,s::BlockSnapshots
      ) -> (RBSpace, RBSpace)

Computes the subspace of the test, trial FE spaces contained in the FE operator
`feop` by compressing the snapshots `s`

"""
function reduced_fe_space(solver::RBSolver,feop,s)
  red = get_state_reduction(solver)
  soff = select_snapshots(s,offline_params(solver))
  reduced_fe_space(red,feop,soff)
end

function reduced_fe_space(red::Reduction,feop,s)
  t = @timed begin
    basis = reduced_basis(red,feop,s)
  end
  println(CostTracker(t,name="Basis construction"))

  reduced_trial = fe_subspace(get_trial(feop),basis)
  reduced_test = fe_subspace(get_test(feop),basis)
  return reduced_trial,reduced_test
end

"""
    reduced_basis(red,s::AbstractSnapshots,args...) -> (Projection, Projection)

Computes the bases spanning the subspace of test, trial FE spaces by compressing
the snapshots `s`

"""
function reduced_basis(
  red::Reduction,
  s::AbstractArray,
  args...)

  projection(red,s,args...)
end

function reduced_basis(
  red::Reduction,
  feop::GridapType,
  s::AbstractArray)

  reduced_basis(red,s)
end

function reduced_basis(
  red::Reduction{<:ReductionStyle,EnergyNorm},
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
  supr_matrix = assemble_matrix(get_supr(red),get_trial(feop),get_test(feop))
  basis = reduced_basis(get_reduction(red),s,norm_matrix)
  enrich!(red,basis,norm_matrix,supr_matrix)
  return basis
end

function fe_subspace(space::FESpace,basis)
  @abstractmethod
end

"""
    abstract type RBSpace <: FESpace end

Represents a vector subspace of a FE space.

Subtypes:
- [`SingleFieldRBSpace`](@ref)
- [`MultiFieldRBSpace`](@ref)

"""
abstract type RBSpace <: FESpace end

(U::RBSpace)(μ) = evaluate(U,μ)

get_fe_space(r::RBSpace) = @abstractmethod
get_reduced_subspace(r::RBSpace) = @abstractmethod

get_basis(r::RBSpace) = get_basis(get_reduced_subspace(r))
num_fe_dofs(r::RBSpace) = num_fe_dofs(get_fe_space(r))

FESpaces.num_free_dofs(r::RBSpace) = num_reduced_dofs(get_reduced_subspace(r))

FESpaces.get_free_dof_ids(r::RBSpace) = Base.OneTo(num_free_dofs(r))

FESpaces.get_vector_type(r::RBSpace) = get_vector_type(get_fe_space(r))

function Algebra.allocate_in_domain(r::RBSpace)
  zero_free_values(r)
end

function Algebra.allocate_in_range(r::RBSpace)
  zero_free_values(get_fe_space(r))
end

function Arrays.return_cache(::typeof(project),r::RBSpace,x::AbstractVector)
  allocate_in_domain(r)
end

function Arrays.return_cache(::typeof(inv_project),r::RBSpace,x::AbstractVector)
  allocate_in_range(r)
end

for (f,f!) in zip((:project,:inv_project),(:project!,:inv_project!))
  @eval begin
    function $f(r::RBSpace,x::AbstractVector)
      y = return_cache($f,r,x)
      $f!(y,r,x)
      return y
    end

    function $f!(y,r::RBSpace,x::AbstractVector)
      y .= $f(get_reduced_subspace(r),x)
      return y
    end

    function $f!(y,r::RBSpace,x::AbstractParamVector)
      @inbounds for ip in eachindex(x)
        y[ip] = $f(get_reduced_subspace(r),x[ip])
      end
      return y
    end
  end
end

function project(r::RBSpace,x::Projection)
  galerkin_projection(get_reduced_subspace(r),x)
end

function project(r1::RBSpace,x::Projection,r2::RBSpace)
  galerkin_projection(get_reduced_subspace(r1),x,get_reduced_subspace(r2))
end

function FESpaces.FEFunction(r::RBSpace,x̂::AbstractVector)
  x = inv_project(r,x̂)
  fe = get_fe_space(r)
  xdir = get_dirichlet_values(fe)
  return FEFunction(fe,x,xdir)
end

"""
    SingleFieldRBSpace{A<:SingleFieldFESpace,B<:Projection} <: RBSpace

Reduced basis subspace in a steady setting

"""
struct SingleFieldRBSpace{A<:SingleFieldFESpace,B<:Projection} <: RBSpace
  space::A
  subspace::B
end

function fe_subspace(space::SingleFieldFESpace,basis::Projection)
  SingleFieldRBSpace(space,basis)
end

get_fe_space(r::SingleFieldRBSpace) = r.space
get_reduced_subspace(r::SingleFieldRBSpace) = r.subspace

"""
    MultiFieldRBSpace{A<:MultiFieldFESpace,B<:BlockProjection} <: RBSpace

Reduced basis subspace in a MultiField setting

"""
struct MultiFieldRBSpace{A<:MultiFieldFESpace,B<:BlockProjection} <: RBSpace
  space::A
  subspace::B
end

function fe_subspace(space::MultiFieldFESpace,subspace::BlockProjection)
  MultiFieldRBSpace(space,subspace)
end

get_fe_space(r::MultiFieldRBSpace) = r.space
get_reduced_subspace(r::MultiFieldRBSpace) = r.subspace

function Base.getindex(r::MultiFieldRBSpace,i::Integer)
  return fe_subspace(r.space.spaces[i],r.subspace[i])
end

function Base.iterate(r::MultiFieldRBSpace,i)
  space = iterate(r.space.spaces,i)
  if isnothing(space)
    return
  end
  ri = fe_subspace(space,r.subspace[i])
  return ri,i+1
end

for f in (:project,:inv_project)
  @eval begin
    function Arrays.return_cache(::typeof($f),r::MultiFieldRBSpace,x::Union{BlockVector,BlockParamVector})
      cache = return_cache($f,r[1],x[Block(1)])
      block_cache = Vector{typeof(cache)}(undef,num_fields(r))
      return mortar(block_cache)
    end
  end
end

# dealing with the transient case here

function Arrays.evaluate(r::RBSpace,args...)
  space = evaluate(get_fe_space(r),args...)
  subspace = fe_subspace(space,get_reduced_subspace(r))
  EvalRBSpace(subspace,args...)
end

struct EvalRBSpace{A<:RBSpace,B<:AbstractRealization} <: RBSpace
  subspace::A
  realization::B
end

get_fe_space(r::EvalRBSpace) = get_fe_space(r.subspace)
get_reduced_subspace(r::EvalRBSpace) = get_reduced_subspace(r.subspace)

FESpaces.get_free_dof_ids(r::EvalRBSpace) = get_free_dof_ids(r.subspace)

const EvalMultiFieldRBSpace{B<:AbstractRealization} = EvalRBSpace{<:MultiFieldRBSpace,B}

function Base.getindex(r::EvalMultiFieldRBSpace,i)
  spacei = getindex(r.subspace,i)
  EvalRBSpace(spacei,r.realization)
end

function Base.iterate(r::EvalMultiFieldRBSpace,i)
  subspace = iterate(r.subspace,i)
  if isnothing(subspace)
    return
  end
  ri = EvalRBSpace(subspace,r.realization)
  return ri,i+1
end

for T in (:MultiFieldRBSpace,:EvalMultiFieldRBSpace)
  @eval begin

    MultiField.MultiFieldStyle(r::$T) = MultiFieldStyle(get_fe_space(r))
    MultiField.num_fields(r::$T) = num_fields(get_fe_space(r))

    function FESpaces.get_free_dof_ids(r::$T)
      get_free_dof_ids(r,MultiFieldStyle(r))
    end

    function FESpaces.get_free_dof_ids(r::$T,::ConsecutiveMultiFieldStyle)
      @notimplemented
    end

    function FESpaces.get_free_dof_ids(r::$T,::BlockMultiFieldStyle{NB}) where NB
      block_num_dofs = map(range->num_free_dofs(r[range]),1:NB)
      return BlockArrays.blockedrange(block_num_dofs)
    end
  end
end

for (S,T) in zip((:BlockVector,:BlockParamVector),(:MultiFieldRBSpace,:EvalMultiFieldRBSpace))
  for f! in (:project!,:inv_project!)
    @eval begin
      function $f!(
        y::$S,
        r::$T,
        x::$S)

        for i in 1:blocklength(x)
          $f!(y[Block(i)],r[i],x[Block(i)])
        end
        return y
      end
    end
  end
end
