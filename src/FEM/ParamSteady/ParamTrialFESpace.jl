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

function get_vector_index_map(f::UnEvalParamSingleFieldFESpace)
  get_vector_index_map(f.space)
end

function get_matrix_index_map(f::UnEvalParamSingleFieldFESpace,g::SingleFieldFESpace)
  get_matrix_index_map(f.space,g)
end

struct ParamTrialESpace{A,B} <: UnEvalParamSingleFieldFESpace
  space::A
  space0::B
  dirichlet::Union{Function,AbstractVector{<:Function}}

  function ParamTrialESpace(
    space::A,
    dirichlet::Union{Function,AbstractVector{<:Function}}) where A

    space0 = HomogeneousTrialFESpace(space)
    B = typeof(space0)
    new{A,B}(space,space0,dirichlet)
  end
end

function ParamTrialESpace(space)
  HomogeneousTrialFESpace(space)
end

FESpaces.ConstraintStyle(::Type{<:ParamTrialESpace{A}}) where A = ConstraintStyle(A)

function ODEs.allocate_space(U::ParamTrialESpace,params)
  HomogeneousTrialParamFESpace(U.space,Val(length(params)))
end

function Arrays.evaluate!(Upt::TrialParamFESpace,U::ParamTrialESpace,params)
  dir(f) = f(params)
  dir(f::Vector) = dir.(f)
  TrialParamFESpace!(Upt,dir(U.dirichlet))
  Upt
end

function Arrays.evaluate!(Upt::TrialParamFESpace,U::ParamTrialESpace,r::ParamRealization)
  evaluate!(Upt,U,get_params(r))
end

Arrays.evaluate(U::ParamTrialESpace,r::Nothing) = U.space0

# Define the ParamTrialESpace interface for stationary spaces

ODEs.allocate_space(U::FESpace,r) = U
Arrays.evaluate!(Upt::FESpace,U::FESpace,r) = U
Arrays.evaluate(U::FESpace,r) = U

# Define the interface for MultiField

const ParamMultiFieldFESpace = MultiFieldFESpace

function has_param(U::MultiFieldFESpace)
  any(space -> space isa ParamTrialESpace,U.spaces)
end

function ODEs.allocate_space(U::MultiFieldFESpace,r::ParamRealization)
  if !has_param(U)
    return U
  end
  spaces = map(U->allocate_space(U,r),U.spaces)
  style = MultiFieldStyle(U)
  MultiFieldParamFESpace(spaces;style)
end

function Arrays.evaluate!(Upt::MultiFieldFESpace,U::MultiFieldFESpace,r::ParamRealization)
  if !has_param(U)
    return U
  end
  for (Upti,Ui) in zip(Upt,U)
    evaluate!(Upti,Ui,r)
  end
  Upt
end

function Arrays.evaluate(U::MultiFieldFESpace,args...)
  Upt = allocate_space(U,args...)
  evaluate!(Upt,U,args...)
  Upt
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
