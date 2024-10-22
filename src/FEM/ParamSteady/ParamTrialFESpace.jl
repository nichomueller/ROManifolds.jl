"""
    abstract type UnEvalParamSingleFieldFESpace <: SingleFieldFESpace end

Represents a trial FESpace with a function representing the Dirichlet boundary
condition, instead of a vector of values. When evaluating an instance of
UnEvalParamSingleFieldFESpace in a parametric realization, a parametric trial
FE space is returned

Subtypes:
- [`ParamTrialFESpace`](@ref)
- [`TransientTrialParamFESpace`](@ref)

"""
abstract type UnEvalParamSingleFieldFESpace <: SingleFieldFESpace end

function Arrays.evaluate(U::UnEvalParamSingleFieldFESpace,args...)
  Upt = allocate_space(U,args...)
  evaluate!(Upt,U,args...)
  Upt
end

(U::UnEvalParamSingleFieldFESpace)(r) = evaluate(U,r)

FESpaces.get_free_dof_ids(f::UnEvalParamSingleFieldFESpace) = get_free_dof_ids(f.space)
FESpaces.get_vector_type(f::UnEvalParamSingleFieldFESpace) = get_vector_type(f.space)
CellData.get_triangulation(f::UnEvalParamSingleFieldFESpace) = get_triangulation(f.space)
FESpaces.get_cell_dof_ids(f::UnEvalParamSingleFieldFESpace) = get_cell_dof_ids(f.space)
FESpaces.get_fe_basis(f::UnEvalParamSingleFieldFESpace) = get_fe_basis(f.space)
FESpaces.get_fe_dof_basis(f::UnEvalParamSingleFieldFESpace) = get_fe_dof_basis(f.space)
function FESpaces.get_cell_constraints(f::UnEvalParamSingleFieldFESpace,c::Constrained)
  get_cell_constraints(f.space,c)
end
function FESpaces.get_cell_isconstrained(f::UnEvalParamSingleFieldFESpace,c::Constrained)
  get_cell_isconstrained(f.space,c)
end

FESpaces.get_dirichlet_dof_ids(f::UnEvalParamSingleFieldFESpace) = get_dirichlet_dof_ids(f.space)
FESpaces.num_dirichlet_tags(f::UnEvalParamSingleFieldFESpace) = num_dirichlet_tags(f.space)
FESpaces.get_dirichlet_dof_tag(f::UnEvalParamSingleFieldFESpace) = get_dirichlet_dof_tag(f.space)
function FESpaces.scatter_free_and_dirichlet_values(f::UnEvalParamSingleFieldFESpace,free_values,dirichlet_values)
  scatter_free_and_dirichlet_values(f.space,free_values,dirichlet_values)
end
function FESpaces.gather_free_and_dirichlet_values!(free_values,dirichlet_values,f::UnEvalParamSingleFieldFESpace,cell_vals)
  gather_free_and_dirichlet_values!(free_values,dirichlet_values,f.space,cell_vals)
end

function FESpaces.get_dirichlet_dof_values(f::UnEvalParamSingleFieldFESpace)
  msg = """
  It does not make sense to get the Dirichlet DOF values of a transient FE space. You
  should first evaluate the transient FE space at a point in time and get the Dirichlet
  DOF values from there.
  """
  @unreachable msg
end

"""
    struct ParamTrialFESpace{A,B} <: UnEvalParamSingleFieldFESpace end

Structure used in steady applications. When a ParamTrialFESpace is evaluated in a
[`Realization`](@ref), a parametric trial FE space is returned

"""
struct ParamTrialFESpace{A,B} <: UnEvalParamSingleFieldFESpace
  space::A
  space0::B
  dirichlet::Union{Function,AbstractVector{<:Function}}

  function ParamTrialFESpace(
    space::A,
    dirichlet::Union{Function,AbstractVector{<:Function}}) where A

    space0 = HomogeneousTrialFESpace(space)
    B = typeof(space0)
    new{A,B}(space,space0,dirichlet)
  end
end

function ParamTrialFESpace(space)
  HomogeneousTrialFESpace(space)
end

FESpaces.ConstraintStyle(::Type{<:ParamTrialFESpace{A}}) where A = ConstraintStyle(A)

function ODEs.allocate_space(U::ParamTrialFESpace,r::Realization)
  HomogeneousTrialParamFESpace(U.space,Val(length(r)))
end

function Arrays.evaluate!(Upt::TrialParamFESpace,U::ParamTrialFESpace,r::Realization)
  dir(f) = f(r)
  dir(f::Vector) = dir.(f)
  TrialParamFESpace!(Upt,dir(U.dirichlet))
  Upt
end

Arrays.evaluate(U::ParamTrialFESpace,r::Nothing) = U.space0

# Define the ParamTrialFESpace interface for stationary spaces

ODEs.allocate_space(U::FESpace,r) = U
Arrays.evaluate!(Upt::FESpace,U::FESpace,r::AbstractRealization) = U
Arrays.evaluate(U::FESpace,r) = U

# Define the interface for MultiField

const ParamMultiFieldFESpace = MultiFieldFESpace

function has_param(U::MultiFieldFESpace)
  any(space -> space isa ParamTrialFESpace,U.spaces)
end

function ODEs.allocate_space(U::MultiFieldFESpace,r::Realization)
  if !has_param(U)
    return U
  end
  spaces = map(U->allocate_space(U,r),U.spaces)
  style = MultiFieldStyle(U)
  MultiFieldParamFESpace(spaces;style)
end

function Arrays.evaluate!(Upt::MultiFieldFESpace,U::MultiFieldFESpace,r::Realization)
  if !has_param(U)
    return U
  end
  for (Upti,Ui) in zip(Upt,U)
    evaluate!(Upti,Ui,r)
  end
  Upt
end

function Arrays.evaluate(U::MultiFieldFESpace,r::Realization)
  if !has_param(U)
    return U
  end
  Ut = allocate_space(U,r)
  evaluate!(Ut,U,r)
  Ut
end

function test_param_trial_fe_space(Uh,μ)
  UhX = evaluate(Uh,nothing)
  @test isa(UhX,FESpace)
  Uh0 = allocate_space(Uh,μ)
  Uh0 = evaluate!(Uh0,Uh,μ)
  @test isa(Uh0,FESpace)
  Uh0 = evaluate(Uh,μ)
  @test isa(Uh0,FESpace)
  Uh0 = Uh(μ)
  @test isa(Uh0,FESpace)
  Uht=∂t(Uh)
  Uht0=Uht(μ)
  @test isa(Uht0,FESpace)
  true
end
