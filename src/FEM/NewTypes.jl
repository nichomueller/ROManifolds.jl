abstract type ParameterType end

abstract type SamplingStyle end
struct Uniform <: FunctionalStyle end
struct Gaussian <: FunctionalStyle end
struct LatinHypercube <: FunctionalStyle end

mutable struct ParameterSpace{PT<:ParameterType}
  domain::PT
  sampling_style::SamplingStyle
end

# for navier stokes
mutable struct ComposedParameterSpace{PT<:ParameterType}
  param_space::ParameterSpace{PT}
  composing_function::Function
end

realization(ss::SamplingStyle) = rand(MersenneTwister(1234), ss)
realization(ss::SamplingStyle, n::Int) = Broadcasting(parameter(ss))(1:n)

function realization(P::ParameterSpace{Vector{Vector{T}}}) where T
  dim = length(P.domain)
  realization(P.sampling_style, dim)
end



abstract type FunctionalStyle end
struct Affine <: FunctionalStyle end
struct NonAffine <: FunctionalStyle end
struct NonLinear <: FunctionalStyle end

mutable struct ParametricFunctional{FS<:FunctionalStyle,PT<:ParameterType,S<:Bool}
  param_domain::ParameterSpace{PT}
  f::Function
end

function ParametricFunctional(
  P::ParameterSpace{PT},
  f::Function;
  FS=NonAffine(),S=true) where {PT}

  ParametricFunctional{FS,PT,S}(P,f)
end

function _get_functional(
  Fμ::ParametricFunctional{FS,PT,true}) where {FS,PT}

  f(μ,x) = Fμ.f(μ,x)
  f(μ) = x -> f(μ,x)
  f
end

function _get_functional(
  Fμ::ParametricFunctional{FS,PT,false}) where {FS,PT}

  f(μ,t,x) = Fμ.f(μ,t,x)
  f(μ,t) = x -> f(μ,t,x)
  f(μ) = t -> f(μ,t)
  f
end

function realization(
  Fμ::ParametricFunctional{FS,PT,S}) where {FS,PT,S}

  _get_functional(Fμ)(realization(Fμ.param_domain))
end

function realization(
  P::ParameterSpace{PT},
  f::Function;
  FS=NonAffine(),S=true) where {PT}

  Fμ = ParametricFunctional(P,f;FS,S)
  realization(Fμ)
end

abstract type FEFunctional{S<:Bool,DBC<:Bool,N<:Int}  end

mutable struct LinFEFunctional{S} <: FEFunctional{S,false,1}
  measure::Measure
  test::UnconstrainedFESpace
  f::Function
end

function LinFEFunctional(
  dΩ::Measure,
  V::UnconstrainedFESpace,
  f::Function;S=true)

  LinFEFunctional{S}(dΩ,V,f)
end

mutable struct BilinFEFunctional{S,DBC} <: FEFunctional{S,DBC,2}
  measure::Measure
  trial::MyTrialFESpace{DBC}
  test::UnconstrainedFESpace
  f::Function
end

function BilinFEFunctional(
  dΩ::Measure,
  U::MyTrialFESpace{DBC},
  V::UnconstrainedFESpace,
  f::Function;S=true)

  BilinFEFunctional{S,DBC}(dΩ,U,V,f)
end

abstract type ParamFEFunctional{FS<:FunctionalStyle,PT<:ParameterType,S<:Bool,DBC<:Bool} end

mutable struct ParamLinFEFunctional{FS,PT,S} <: FEFunctional{FS,PT,S,false}
  fe_functional::LinFEFunctional{S}
  param_functional::ParametricFunctional{FS,PT,S}
end

mutable struct ParamBilinFEFunctional{FS,PT,S,DBC} <: FEFunctional{FS,PT,S,DBC}
  fe_functional::BilinFEFunctional{S,DBC}
  param_functional::ParametricFunctional{FS,PT,S}
end

function ParamFEFunctional(
  Fv::FEFunctional{S,false,1},
  Fμ::ParametricFunctional{FS,PT,S}) where {FS,PT,S}

  ParamLinFEFunctional{FS,PT,S}(Fv,Fμ)
end

function ParamFEFunctional(
  Fuv::FEFunctional{S,DBC,2},
  Fμ::ParametricFunctional{FS,PT,S}) where {FS,PT,S,DBC}

  ParamLinFEFunctional{FS,PT,S,DBC}(Fuv,Fμ)
end

function _compose_functionals(
  Fuv::FEFunctional{true,DBC,N},
  Fμ::ParametricFunctional{FS,PT,true}) where {FS,PT,DBC,N}

  form(μ,args...) = ∫(Fuv.f(Fμ(μ),args...))Fuv.measure
  form(μ) = args -> form(μ,args...)
  form
end

function _compose_functionals(
  Fuv::FEFunctional{false,DBC,N},
  Fμ::ParametricFunctional{FS,PT,false}) where {FS,PT,DBC,N}

  args = (N == 2) ? (v) : (u,v)

  form(μ,t,args...) = ∫(Fuv.f(Fμ(μ,t),t,args...))Fuv.measure
  form(μ,t) = args -> form(μ,t,args...)
  form(μ) = t -> form(μ,t)
  form
end

function realization(
  Fuvμ::ParamFEFunctional{FS,PT,S,DBC}) where {FS,PT,S,DBC}

  Fuv, Fμ = Fuvμ.fe_functional, Fuvμ.param_functional
  param_form = _compose_functionals(Fuv, Fμ)
  param_form(realization(Fμ.param_functional))
end

#= function Functional(
  id::String,
  f::Function,
  μ::Parameter;
  FS=NonAffine(), Di=3, Do=3, N=2, S=true)

  @assert N ∈ (1,2)
  Functional{FS,Di,Do,N,S}(id,f,μ)
end

get_id(fun::Functional) = fun.id
get_f(fun::Functional) = fun.f =#

const AffineFunctional{Di,Do,N,S} = Functional{Affine(),Di,Do,N,S}
const NonAffineFunctional{Di,Do,N,S} = Functional{NonAffine(),Di,Do,N,S}
const NonLinearFunctional{Di,Do,N,S} = Functional{NonLinear(),Di,Do,N,S}

function isaffine(::Functional{FS,Di,Do,N,S}) where {FS,Di,Do,N,S}
  FS==Affine() ? true : false
end

function islinear(::Functional{FS,Di,Do,N,S}) where {FS,Di,Do,N,S}
  FS==NonLinear() ? false : true
end

mutable struct FEForm{FS<:FunctionalStyle,Di<:Int,Do<:Int,N<:Int,S<:Bool}
  functional::Functional{FS,Di,Do,N,S}
  trian::Triangulation{Di,Di}
end

abstract type FEQuantity end

abstract type FEArray{FS<:FunctionalStyle,Di<:Int,Do<:Int,N<:Int,S<:Bool,DBC<:Bool} <: FEQuantity end

mutable struct FEVector{FS<:FunctionalStyle,Di<:Int,Do<:Int,S<:Bool} <: FEArray{FS,Di,Do,1,S,False}
  functional::Functional{FS,Di,Do,2,S}
  form::Function
  test::UnconstrainedFESpace
  trian::Triangulation{Di,Di}
  array::Function
end

function FEVector(
  FEMInfo::FOMInfo,
  functional::Functional{FS,Di,Do,1,S},
  form::Function,
  test::UnconstrainedFESpace,
  trian::Triangulation{Di,Di}) where {FS,Di,Do,S}

  m = Measure(trian,FEMInfo.order)
  array = _assemble_array(functional,form,test,m)
  FEVector{FS,Di,Do,S}(functional,form,test,trian,array)
end

mutable struct FEMatrix{FS<:FunctionalStyle,Di<:Int,Do<:Int,S<:Bool,DBC<:Bool} <: FEArray{FS,Di,Do,2,S,DBC}
  functional::Functional{FS,Di,Do,2,S}
  form::Function
  trial_no_bc::UnconstrainedFESpace
  trial::MyFESpace{DBC}
  test::UnconstrainedFESpace
  trian::Triangulation{Di,Di}
  array::Vector{<:Function}
end

function FEMatrix(
  FEMInfo::FOMInfo,
  functional::Functional{FS,Di,Do,2,S},
  form::Function,
  trial_no_bc::UnconstrainedFESpace,
  trial::MyFESpace{DBC},
  test::UnconstrainedFESpace,
  trian::Triangulation{Di,Di}) where {FS,Di,Do,S}

  m = Measure(trian,FEMInfo.order)
  array = _assemble_array(FEMInfo,functional,form,trial_no_bc,trial,test,m)
  FEMatrix{FS,Di,Do,S,DBC}(functional,form,trial_no_bc,trial,test,trian,array)
end

function get_array(arr::FEArray{FS,Di,Do,N,True,DBC}) where {FS,Di,Do,N,DBC}
  μ -> arr.array[1](μ)
end

function get_array(arr::FEArray{FS,Di,Do,N,False,DBC}) where {FS,Di,Do,N,DBC}
  (t,μ) -> arr.array[1](t,μ)
end

function get_array(arr::FEArray{Affine,Di,Do,N,True,DBC}) where {Di,Do,N,DBC}
  arr.array[1](get_μ(arr))
end

function get_array(arr::FEArray{Affine,Di,Do,N,False,DBC}) where {Di,Do,N,DBC}
  t -> arr.array[1](t,get_μ(arr))
end

function get_lift(arr::FEArray{FS,Di,Do,N,S,False}) where {FS,Di,Do,N,S}
  error("No lifting associated to the variable $(get_id(arr))")
end

function get_lift(arr::FEArray{FS,Di,Do,N,True,True}) where {FS,Di,Do,N}
  μ -> arr.array[2](μ)
end

function get_lift(arr::FEArray{FS,Di,Do,N,False,True}) where {FS,Di,Do,N}
  (t,μ) -> arr.array[2](t,μ)
end

function get_lift(arr::FEArray{Affine,Di,Do,N,True,True}) where {Di,Do,N}
  arr.array[2](get_μ(arr))
end

function get_lift(arr::FEArray{Affine,Di,Do,N,False,True}) where {Di,Do,N}
  t -> arr.array[2](t,get_μ(arr))
end

const FEVectorS{FS,Di,Do} = FEVector{FS,Di,Do,true}
const FEVectorST{FS,Di,Do} = FEVector{FS,Di,Do,false}
const FEMatrixS{FS,Di,Do} = FEMatrix{FS,Di,Do,true}
const FEMatrixST{FS,Di,Do} = FEMatrix{FS,Di,Do,false}

const AffineFEVectorS{Di,Do} = FEVectorS{Affine(),Di,Do}
const NonAffineFEVectorS{Di,Do} = FEVectorS{NonAffine(),Di,Do}
const NonLinearFEVectorS{Di,Do} = FEVectorS{NonLinear(),Di,Do}
const AffineFEMatrixS{Di,Do,DBC} = FEMatrixS{Affine(),Di,Do,DBC}
const NonAffineFEMatrixS{Di,Do,DBC} = FEMatrixS{NonAffine(),Di,Do,DBC}
const NonLinearFEMatrixS{Di,Do,DBC} = FEMatrixS{NonLinear(),Di,Do,DBC}

const AffineFEVectorST{Di,Do} = FEVectorST{Affine(),Di,Do}
const NonAffineFEVectorST{Di,Do} = FEVectorST{NonAffine(),Di,Do}
const NonLinearFEVectorST{Di,Do} = FEVectorST{NonLinear(),Di,Do}
const AffineFEMatrixST{Di,Do,DBC} = FEMatrixST{Affine(),Di,Do,DBC}
const NonAffineFEMatrixST{Di,Do,DBC} = FEMatrixST{NonAffine(),Di,Do,DBC}
const NonLinearFEMatrixST{Di,Do,DBC} = FEMatrixST{NonLinear(),Di,Do,DBC}

function _assemble_array(
  functional::Functional{FS,Di,Do,1,true},
  form::Function,
  test::UnconstrainedFESpace,
  m::Measure) where {FS,Di,Do}

  param_form(v,μ) = form(functional(μ),v)
  param_form(μ) = v -> param_form(v,μ)
  μ -> assemble_vector(∫(param_form(μ))m,test)
end

function _assemble_array(
  functional::Functional{FS,Di,Do,1,false},
  form::Function,
  test::UnconstrainedFESpace,
  m::Measure) where {FS,Di,Do}

  param_form(t,v,μ) = form(functional(μ),t,v)
  param_form(t,μ) = v -> param_form(t,v,μ)
  (t,μ) -> assemble_vector(∫(param_form(t,μ))m,test)
end

function _assemble_array(
  FEMInfo::FOMInfo,
  functional::Functional{FS,Di,Do,2,true},
  form::Function,
  trial_no_bc::UnconstrainedFESpace,
  trial::MyFESpace{DBC},
  test::UnconstrainedFESpace,
  m::Measure) where {FS,Di,Do,DBC}

  param_form(u,v,μ) = form(functional(μ),u,v)
  param_form(μ) = (u,v) -> param_form(u,v,μ)
  mat_all_dofs(μ) = assemble_matrix(∫(param_form(μ))m,trial_no_bc,test)
  mat_free_dofs(μ) = mat_all_dofs(μ)[FEMInfo.free_dofs,FEMInfo.free_dofs]
  DBC ? [mat_free_dofs, _assemble_lift(functional,mat_all_dofs,trial)] : [mat_free_dofs]
end

function _assemble_array(
  FEMInfo::FOMInfo,
  functional::Functional{FS,Di,Do,2,true},
  form::Function,
  trial_no_bc::UnconstrainedFESpace,
  trial::MyFESpace{DBC},
  test::UnconstrainedFESpace,
  m::Measure) where {FS,Di,Do,DBC}

  param_form(t,u,v,μ) = form(functional(μ),t,u,v)
  param_form(t,μ) = (u,v) -> param_form(t,u,v,μ)
  mat_all_dofs(t,μ) = assemble_matrix(∫(param_form(t,μ))m,trial_no_bc,test)
  mat_free_dofs(t,μ) = mat_all_dofs(t,μ)[FEMInfo.free_dofs,FEMInfo.free_dofs]
  DBC ? [mat_free_dofs, _assemble_lift(functional,mat_all_dofs,trial)] : [mat_free_dofs]
end

function _assemble_lift(
  ::Functional{FS,Di,Do,2,true},
  mat_all_dofs::Function,
  trial::MyFESpace) where {FS,Di,Do}

  g = trial.dirichlet_values
  mat_all_dofs(μ)[FEMInfo.free_dofs,FEMInfo.dirichlet_dofs]*g

end

function _assemble_lift(
  ::Functional{FS,Di,Do,2,false},
  mat_all_dofs::Function,
  trial::MyFESpace) where {FS,Di,Do}

  g(t) = trial(t).dirichlet_values
  mat_all_dofs(t,μ)[FEMInfo.free_dofs,FEMInfo.dirichlet_dofs]*g(t)

end

get_functional(arr::FEArray) = arr.functional
get_id(arr::FEArray) = arr.fun.id
get_f(arr::FEArray) = arr.fun.f
get_μ(arr::FEArray) = arr.fun.μ
isaffine(arr::FEArray) = isaffine(get_functional(arr))
islinear(arr::FEArray) = islinear(get_functional(arr))
issteady(::FEArray{FS,Di,Do,N,S,DBC}) where {FS,Di,Do,N,S,DBC} = S
has_dirichlet_bc(::FEArray{FS,Di,Do,N,S,DBC}) where {FS,Di,Do,N,S,DBC} = DBC

struct FEProblem{Di,Do,S,DBC,Ind}
  ids::Vector{String}
  fs::Vector{Function}
  μs::Vector{Parameter}
  problem::Vector{<:FEArray}
end
