struct TransientTrialParamFESpace{A,B} <: SingleFieldFESpace
  space::A
  space0::B
  dirichlet::Union{Function,AbstractVector{<:Function}}

  function TransientTrialParamFESpace(
    space::A,
    dirichlet::Union{Function,AbstractVector{<:Function}}) where A

    space0 = HomogeneousTrialFESpace(space)
    B = typeof(space0)
    new{A,B}(space,space0,dirichlet)
  end
end

function TransientTrialParamFESpace(space)
  HomogeneousTrialFESpace(space)
end

function ODEs.allocate_space(U::TransientTrialParamFESpace,params,times)
  HomogeneousTrialParamFESpace(U.space,Val(length(params)*length(times)))
end

function ODEs.allocate_space(U::TransientTrialParamFESpace,r::TransientParamRealization)
  allocate_space(U,get_params(r),get_times(r))
end

function Arrays.evaluate!(
  Upt::TrialParamFESpace,
  U::TransientTrialParamFESpace,
  params,
  times)

  dir(f) = f(params,times)
  dir(f::Vector) = dir.(f)
  TrialParamFESpace!(Upt,dir(U.dirichlet))
  Upt
end

function Arrays.evaluate!(
  Upt::TrialParamFESpace,
  U::TransientTrialParamFESpace,
  r::TransientParamRealization)

  evaluate!(Upt,U,get_params(r),get_times(r))
end

function Arrays.evaluate(U::TransientTrialParamFESpace,args...)
  Upt = allocate_space(U,args...)
  evaluate!(Upt,U,args...)
  Upt
end

Arrays.evaluate(U::TransientTrialParamFESpace,params::Nothing,times::Nothing) = U.space0
Arrays.evaluate(U::TransientTrialParamFESpace,r::Nothing) = U.space0

(U::TransientTrialParamFESpace)(params,times) = evaluate(U,params,times)
(U::TransientTrialParamFESpace)(r) = evaluate(U,r)
(U::TrialFESpace)(params,times) = U
(U::ZeroMeanFESpace)(params,times) = U

function ODEs.time_derivative(U::TransientTrialParamFESpace)
  ∂tdir(f) = (μ,t) -> time_derivative(f(μ,t))
  ∂tdir(f::Vector) = ∂tdir.(f)
  TransientTrialParamFESpace(U.space,∂tdir(U.dirichlet))
end

FESpaces.get_free_dof_ids(f::TransientTrialParamFESpace) = get_free_dof_ids(f.space)
FESpaces.get_vector_type(f::TransientTrialParamFESpace) = get_vector_type(f.space)
CellData.get_triangulation(f::TransientTrialParamFESpace) = get_triangulation(f.space)
FESpaces.get_cell_dof_ids(f::TransientTrialParamFESpace) = get_cell_dof_ids(f.space)
FESpaces.get_fe_basis(f::TransientTrialParamFESpace) = get_fe_basis(f.space)
FESpaces.get_fe_dof_basis(f::TransientTrialParamFESpace) = get_fe_dof_basis(f.space)
FESpaces.ConstraintStyle(::Type{<:TransientTrialParamFESpace{U}}) where U = ConstraintStyle(U)
function FESpaces.get_cell_constraints(f::TransientTrialParamFESpace,c::Constrained)
  get_cell_constraints(f.space,c)
end
function FESpaces.get_cell_isconstrained(f::TransientTrialParamFESpace,c::Constrained)
  get_cell_isconstrained(f.space,c)
end

FESpaces.get_dirichlet_dof_ids(f::TransientTrialParamFESpace) = get_dirichlet_dof_ids(f.space)
FESpaces.num_dirichlet_tags(f::TransientTrialParamFESpace) = num_dirichlet_tags(f.space)
FESpaces.get_dirichlet_dof_tag(f::TransientTrialParamFESpace) = get_dirichlet_dof_tag(f.space)
function FESpaces.scatter_free_and_dirichlet_values(f::TransientTrialParamFESpace,free_values,dirichlet_values)
  scatter_free_and_dirichlet_values(f.space,free_values,dirichlet_values)
end
function FESpaces.gather_free_and_dirichlet_values!(free_values,dirichlet_values,f::TransientTrialParamFESpace,cell_vals)
  gather_free_and_dirichlet_values!(free_values,dirichlet_values,f.space,cell_vals)
end

function FESpaces.get_dirichlet_dof_values(f::TransientTrialParamFESpace)
  msg = """
  It does not make sense to get the Dirichlet DOF values of a transient FE space. You
  should first evaluate the transient FE space at a point in time and get the Dirichlet
  DOF values from there.
  """
  @unreachable msg
end

function TProduct.get_vector_index_map(f::TransientTrialParamFESpace)
  get_vector_index_map(f.space)
end

function TProduct.get_matrix_index_map(f::TransientTrialParamFESpace,g::SingleFieldFESpace)
  get_matrix_index_map(f.space,g)
end

# Define the TransientTrialFESpace interface for stationary spaces

ODEs.allocate_space(U::FESpace,params,times) = U
ODEs.allocate_space(U::FESpace,r) = U
Arrays.evaluate!(Upt::FESpace,U::FESpace,params,times) = U
Arrays.evaluate!(Upt::FESpace,U::FESpace,r) = U
Arrays.evaluate(U::FESpace,params,times) = U
Arrays.evaluate(U::FESpace,r) = U
(space::FESpace)(params,times) = evaluate(space,params,times)

# Define the interface for MultiField

const TransientMultiFieldParamFESpace = MultiFieldFESpace

function has_param_transient(U::MultiFieldFESpace)
  any(space -> space isa TransientTrialParamFESpace,U.spaces)
end

function ODEs.allocate_space(U::MultiFieldFESpace,args...)
  if !has_param_transient(U)
    @assert !ODEs.has_transient(U)
    return U
  end
  spaces = map(U->allocate_space(U,args...),U.spaces)
  style = MultiFieldStyle(U)
  MultiFieldParamFESpace(spaces;style)
end

function Arrays.evaluate!(
  Upt::MultiFieldFESpace,
  U::MultiFieldFESpace,
  args...)

  if !has_param_transient(U)
    @assert !ODEs.has_transient(U)
    return Ut
  end
  for (Upti,Ui) in zip(Upt,U)
    evaluate!(Upti,Ui,args...)
  end
  Upt
end

function Arrays.evaluate(U::MultiFieldFESpace,args...)
  Upt = allocate_space(U,args...)
  evaluate!(Upt,U,args...)
  Upt
end

function test_transient_trial_fe_space(Uh,μ)
  UhX = evaluate(Uh,nothing)
  @test isa(UhX,FESpace)
  Uh0 = allocate_space(Uh,μ,0.0)
  Uh0 = evaluate!(Uh0,Uh,μ,0.0)
  @test isa(Uh0,FESpace)
  Uh0 = evaluate(Uh,μ,0.0)
  @test isa(Uh0,FESpace)
  Uh0 = Uh(μ,0.0)
  @test isa(Uh0,FESpace)
  Uht=∂t(Uh)
  Uht0=Uht(μ,0.0)
  @test isa(Uht0,FESpace)
  true
end
