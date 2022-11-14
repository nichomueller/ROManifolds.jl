abstract type ParameterType end

abstract type SamplingStyle end
struct Uniform <: SamplingStyle end
struct Gaussian <: SamplingStyle end
struct LatinHypercube <: SamplingStyle end

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

mutable struct ParametricFunctional{FS<:FunctionalStyle,PT<:ParameterType,S}
  param_space::ParameterSpace{PT}
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

  μ = realization(Fμ.param_space)
  μ, _get_functional(Fμ)(μ)
end

function realization(
  P::ParameterSpace{PT},
  f::Function;
  FS=NonAffine(),S=true) where {PT}

  Fμ = ParametricFunctional(P,f;FS,S)
  realization(Fμ)
end

abstract type MyFESpace{DBC} end

struct MyTestFESpace <: MyFESpace{false}
  space::UnconstrainedFESpace
  space_no_bc::UnconstrainedFESpace
end

struct MyTrialFESpace <: MyFESpace{true}
  space::ParamTrialFESpace
  space_no_bc::UnconstrainedFESpace
end

struct MyTransientTrialFESpace <: MyFESpace{true}
  space::ParamTransientTrialFESpace
  space_no_bc::UnconstrainedFESpace
end

function get_fespace_no_bc(reffe,model)
  test_no_bnd = FESpace(model,reffe)
  trial_no_bnd = TrialFESpace(test_no_bnd)
  test_no_bnd,trial_no_bnd
end

function MyFESpace(
  space::UnconstrainedFESpace,
  reffe::Tuple,
  model::DiscreteModel)

  space_no_bc = get_fespace_no_bc(reffe,model)
  MyTestFESpace(space,space_no_bc)
end

function MyFESpace(
  space::TrialFESpace,
  reffe::Tuple,
  model::DiscreteModel)

  space_no_bc = get_fespace_no_bc(reffe,model)
  MyTrialFESpace(space,space_no_bc)
end

function MyFESpace(
  space::TransientTrialFESpace,
  reffe::Tuple,
  model::DiscreteModel)

  space_no_bc = get_fespace_no_bc(reffe,model)
  MyTransientTrialFESpace(space,space_no_bc)
end

abstract type FEFunctional{N,DBC}  end

mutable struct LinFEFunctional <: FEFunctional{1,false}
  measure::Measure
  test::MyTestFESpace
  f::Function
end

function LinFEFunctional(
  dΩ::Measure,
  V::UnconstrainedFESpace,
  f::Function)

  LinFEFunctional(dΩ,MyFESpace(V),f)
end

mutable struct BilinFEFunctional{DBC} <: FEFunctional{2,DBC}
  measure::Measure
  trial::MyFESpace{DBC}
  test::MyTestFESpace
  f::Function
end

function BilinFEFunctional(
  dΩ::Measure,
  U::GridapType,
  V::UnconstrainedFESpace,
  f::Function)

  BilinFEFunctional{true}(dΩ,MyFESpace(U),MyFESpace(V),f)
end

function BilinFEFunctional(
  dΩ::Measure,
  U::UnconstrainedFESpace,
  V::UnconstrainedFESpace,
  f::Function)

  BilinFEFunctional{false}(dΩ,MyFESpace(U),MyFESpace(V),f)
end

abstract type ParamFEQuantity{FS<:FunctionalStyle,PT<:ParameterType,S,DBC} end
abstract type ParamFEFunctional{FS<:FunctionalStyle,PT<:ParameterType,S,DBC} <: ParamFEQuantity{FS,PT,S,DBC} end

mutable struct ParamLinFEFunctional{FS,PT,S} <: ParamFEFunctional{FS,PT,S,false}
  param_functional::ParametricFunctional{FS,PT,S}
  fe_functional::LinFEFunctional
end

mutable struct ParamBilinFEFunctional{FS,PT,S,DBC} <: ParamFEFunctional{FS,PT,S,DBC}
  param_functional::ParametricFunctional{FS,PT,S}
  fe_functional::BilinFEFunctional{DBC}
end

function ParamFEFunctional(
  Fμ::ParametricFunctional{FS,PT,S},
  Fv::LinFEFunctional) where {FS,PT,S}

  ParamLinFEFunctional{FS,PT,S}(Fμ,Fv)
end

function ParamFEFunctional(
  Fμ::ParametricFunctional{FS,PT,S},
  Fuv::BilinFEFunctional{DBC}) where {FS,PT,S,DBC}

  ParamBilinFEFunctional{FS,PT,S,DBC}(Fμ,Fuv)
end

function _compose_functionals(
  Fμ::ParametricFunctional{FS,PT,true},
  Fuv::FEFunctional{N,DBC}) where {FS,PT,N,DBC}

  form(μ,args...) = ∫(Fuv.f(Fμ(μ),args...))Fuv.measure
  form(μ) = args -> form(μ,args...)
  form
end

function _compose_functionals(
  Fμ::ParametricFunctional{FS,PT,false},
  Fuv::FEFunctional{N,DBC}) where {FS,PT,N,DBC}

  args = (N == 2) ? (v) : (u,v)

  form(μ,t,args...) = ∫(Fuv.f(Fμ(μ,t),t,args...))Fuv.measure
  form(μ,t) = args -> form(μ,t,args...)
  form(μ) = t -> form(μ,t)
  form
end

function realization(
  Fuvμ::ParamFEFunctional{FS,PT,S,DBC}) where {FS,PT,S,DBC}

  Fuv, Fμ = Fuvμ.fe_functional, Fuvμ.param_functional
  param_form = _compose_functionals(Fμ,Fuv)
  μ, Fμ_μ = realization(Fμ.param_functional)
  μ, param_form(Fμ_μ)
end

abstract type FEArray{FS<:FunctionalStyle,PT<:ParameterType,N,S,DBC} <: ParamFEQuantity{FS,PT,S,DBC} end

mutable struct FEVector{FS<:FunctionalStyle,PT<:ParameterType,S} <: FEArray{FS,PT,1,S,false}
  id::String
  param_fe_functional::ParamLinFEFunctional{FS,PT,S}
  array::Function
end

mutable struct FEMatrix{FS<:FunctionalStyle,PT<:ParameterType,S,DBC} <: FEArray{FS,PT,2,S,DBC}
  id::String
  param_fe_functional::ParamBilinFEFunctional{FS,PT,S,DBC}
  array::Vector{<:Function}
end

function Base.getproperty(fe_array::FEArray, sym::Symbol)
  if sym ∈ (param_functional,fe_functional)
    getfield(fe_array.param_fe_functional, sym)
  else
    getfield(fe_array, sym)
  end
end

function Base.setproperty!(fe_array::FEArray, sym::Symbol, x)
  if sym ∈ (param_functional,fe_functional)
    setfield!(fe_array.param_fe_functional, sym, x)
  else
    setfield!(fe_array, sym, x)
  end
end

function FEArray(
  id::String,
  Fvμ::ParamLinFEFunctional{FS,PT,S}) where {FS,PT,S}

  FEVector{FS,PT,S}(id,Fvμ,_assemble_array(Fvμ))
end

function FEArray(
  id::String,
  Fuvμ::ParamBilinFEFunctional{FS,PT,S,DBC}) where {FS,PT,S,DBC}

  FEMatrix{FS,PT,S,DBC}(id,Fuvμ,_assemble_array(Fuvμ))
end

function FEArray(
  id::String,
  Fμ::ParametricFunctional{FS,PT,S},
  Fuv::FEFunctional{N,DBC}) where {FS,PT,N,S,DBC}

  Fuvμ = ParamFEFunctional(Fμ,Fuv)
  FEArray(id,Fuvμ)
end

function _assemble_array(Fvμ::ParamLinFEFunctional{FS,PT,S}) where {FS,PT,S}

  V = get_test(Fvμ)
  lin_form = _compose_functionals(Fvμ.param_functional,Fvμ.fe_functional)
  μ -> assemble_vector(∫(lin_form(μ))get_measure(Fvμ),V)
end

function _assemble_array(Fuvμ::ParamBilinFEFunctional{FS,PT,S,false}) where {FS,PT,S}

  U,V = get_trial(Fvμ),get_test(Fvμ)
  bilin_form = _compose_functionals(Fuvμ.param_functional,Fuvμ.fe_functional)
  μ -> assemble_matrix(∫(bilin_form(μ))get_measure(Fvμ),U,V)
end

function _assemble_array(Fuvμ::ParamBilinFEFunctional{FS,PT,S,true}) where {FS,PT,S}

  bilin_form = _compose_functionals(Fuvμ.param_functional,Fuvμ.fe_functional)
  mat_all_dofs(μ) = assemble_matrix(∫(bilin_form(μ))get_measure(Fvμ),
    trial.space_no_bc,test.space_no_bc)
  mat_free_dofs(μ) = mat_all_dofs(μ)[trial.space.free_dofs,trial.space.free_dofs]
  [mat_free_dofs, _assemble_lift(mat_all_dofs,trial)]
end

function _assemble_lift(
  mat_all_dofs::Function,
  trial::MyFESpace)

  g(μ) = trial(μ).dirichlet_values
  μ -> mat_all_dofs(μ)[FEMInfo.free_dofs,FEMInfo.dirichlet_dofs]*g(μ)
end

get_trial(q::ParamFEQuantity) = q.fe_functional.trial.space
get_trial_no_bcs(q::ParamFEQuantity) = q.fe_functional.trial.space_no_bc
get_test(q::ParamFEQuantity) = q.fe_functional.test.space
get_test_no_bcs(q::ParamFEQuantity) = q.fe_functional.test.space_no_bc
get_measure(q::ParamFEQuantity) = q.fe_functional.measure
Gridap.get_triangulation(q::ParamFEQuantity) = get_triangulation(get_test(q))

isaffine(::ParamFEQuantity{FS,PT,S,DBC}) where {FS,PT,S,DBC} = (FS == Affine())
islinear(::ParamFEQuantity{FS,PT,S,DBC}) where {FS,PT,S,DBC} = !(FS == NonLinear())
issteady(::ParamFEQuantity{FS,PT,S,DBC}) where {FS,PT,S,DBC} = S
has_dirichlet_bc(::ParamFEQuantity{FS,PT,S,DBC}) where {FS,PT,S,DBC} = DBC

mutable struct ParamFEProblem{D,PT<:ParameterType,Ind,S}
  param_space::ParameterSpace{PT}
  param_fe_functional::Vector{<:ParamFEFunctional}
  param_fe_array::Vector{<:FEArray}

  function ParamFEProblem(
    param_space::ParameterSpace{PT},
    param_fe_functional::Vector{<:ParamFEFunctional},
    param_fe_array::Vector{<:FEArray};
    Ind=true) where PT

    D = num_cell_dims(get_triangulation(first(param_fe_functional)))
    S = issteady(first(param_fe_functional))
    new{D,PT,Ind,S}(param_space,param_fe_functional,param_fe_array)
  end
end

function ParamFEProblem(
  P::ParameterSpace{PT},
  Fμ_dict::Dict{String,ParametricFunctional},
  Ffe_dict::Dict{String,FEFunctional};
  Ind=true) where PT

  @assert keys(Fμ_dict) == keys(Ffe_dict) "Use same variable names"
  id_iter = keys(Fμ_dict)

  Fμ_iter = values(Fμ_dict)
  Ffe_iter = values(Ffe_dict)
  param_fe_funs = Broadcasting(ParamFEFunctional)(Fμ_iter,Ffe_iter)
  param_fe_arrays = Broadcasting(FEArray)(id_iter,param_fe_funs)

  FEProblem{PT,Ind}(P,param_fe_funs,param_fe_arrays)
end
