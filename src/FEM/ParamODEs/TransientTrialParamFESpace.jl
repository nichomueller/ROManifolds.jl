struct TransientTrialParamFESpace{A,B} <: ParametricSingleFieldFESpace
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

Arrays.evaluate(U::TransientTrialParamFESpace,params::Nothing,times::Nothing) = U.space0
Arrays.evaluate(U::TransientTrialParamFESpace,r::Nothing) = U.space0

(U::TransientTrialParamFESpace)(params,times) = evaluate(U,params,times)
(U::TrialFESpace)(params,times) = U
(U::ZeroMeanFESpace)(params,times) = U

function ODEs.time_derivative(U::TransientTrialParamFESpace)
  ∂tdir(f) = (μ,t) -> time_derivative(f(μ,t))
  ∂tdir(f::Vector) = ∂tdir.(f)
  TransientTrialParamFESpace(U.space,∂tdir(U.dirichlet))
end

# Define the TransientTrialParamFESpace interface for stationary spaces

ODEs.allocate_space(U::FESpace,params,times) = U
Arrays.evaluate!(Upt::FESpace,U::FESpace,params,times) = U
Arrays.evaluate(U::FESpace,params,times) = U
(space::FESpace)(params,times) = evaluate(space,params,times)

# Define the interface for MultiField

const TransientMultiFieldParamFESpace = MultiFieldFESpace

function has_param_transient(U::MultiFieldFESpace)
  any(space -> space isa TransientTrialParamFESpace,U.spaces)
end

function ODEs.allocate_space(U::MultiFieldFESpace,μ,t)
  if !has_param_transient(U)
    return U
  end
  spaces = map(U->allocate_space(U,μ,t),U.spaces)
  style = MultiFieldStyle(U)
  MultiFieldParamFESpace(spaces;style)
end

function Arrays.evaluate!(
  Upt::MultiFieldFESpace,
  U::MultiFieldFESpace,
  μ,t)

  if !has_param_transient(U)
    return U
  end
  for (Upti,Ui) in zip(Upt,U)
    evaluate!(Upti,Ui,μ,t)
  end
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
