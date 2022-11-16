abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

mutable struct ParamSpace
  domain::Vector{Vector{Float}}
  sampling_style::SamplingStyle
end

realization(d::Vector{Float},::UniformSampling) = rand(Uniform(first(d),last(d)))
realization(d::Vector{Float},::NormalSampling) = rand(Normal(first(d),last(d)))
realization(P::ParamSpace) = Broadcasting(d->realization(d,P.sampling_style))(P.domain)
realization(P::ParamSpace,n::Int) = [realization(P) for _ = 1:n]

abstract type FunctionalStyle end
struct Affine <: FunctionalStyle end
struct Nonaffine <: FunctionalStyle end
struct Nonlinear <: FunctionalStyle end

mutable struct ParamFunctional{FS,S}
  param_space::ParamSpace
  f::Function

  function ParamFunctional(
    P::ParamSpace,
    f::Function;
    FS=Nonaffine(),S=false)
    new{FS,S}(P,f)
  end
end

function realization(
  Fμ::ParamFunctional{FS,S}) where {FS,S}

  μ = realization(Fμ.param_space)
  μ, Fμ.f(μ)
end

function realization(
  P::ParamSpace,
  f::Function;
  FS=Nonaffine(),S=false)

  Fμ = ParamFunctional(P,f;FS,S)
  realization(Fμ)
end

struct MyTests
  test::UnconstrainedFESpace
  test_no_bc::UnconstrainedFESpace
end

function MyTests(model,reffe;kwargs...)
  test = TestFESpace(model,reffe;kwargs...)
  test_no_bc = FESpace(model,reffe)
  MyTests(test,test_no_bc)
end

struct MyTrial{TT}
  trial::TT
end

function MyTrial(test::UnconstrainedFESpace)
  trial = HomogeneousTrialFESpace(test)
  MyTrial{TrialFESpace}(trial)
end

function MyTrial(
  test::UnconstrainedFESpace,
  Gμ::ParamFunctional{FS,true}) where FS

  trial = ParamTrialFESpace(test,Gμ.f)
  MyTrial{ParamTrialFESpace}(trial)
end

function MyTrial(
  test::UnconstrainedFESpace,
  Gμ::ParamFunctional{FS,false}) where FS

  trial = ParamTransientTrialFESpace(test,Gμ.f)
  MyTrial{ParamTransientTrialFESpace}(trial)
end

struct MyTrials{TT}
  trial::MyTrial{TT}
  trial_no_bc::UnconstrainedFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTrials(
    trial::MyTrial{TT},
    trial_no_bc::UnconstrainedFESpace) where TT

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(trial.trial.space,trial_no_bc)
    new{TT}(trial,trial_no_bc,ddofs_on_full_trian)
  end
end

function MyTrials(test::MyTests,args...)
  trial = MyTrial(test.test,args...)
  trial_no_bc = TrialFESpace(test.test_no_bc)
  MyTrials(trial,trial_no_bc)
end

function free_dofs_on_full_trian(trials::MyTrials)
  nfree_on_full_trian = trials.trial_no_bc.nfree
  setdiff(collect(1:nfree_on_full_trian),trials.ddofs_on_full_trian)
end

abstract type FEFunctional{N,TT}  end

mutable struct LinFEFunctional <: FEFunctional{1,nothing}
  f::Function
  test::MyTests
  measure::Measure
end

mutable struct BilinFEFunctional{TT} <: FEFunctional{2,TT}
  f::Function
  trial::MyTrials{TT}
  test::MyTests
  measure::Measure
end

function FEFunctional(f::Function,spaces::Tuple{MyTests},measure::Measure)
  LinFEFunctional(f,spaces...,measure)
end

function FEFunctional(f::Function,spaces::Tuple{MyTrials,MyTests},measure::Measure)
  BilinFEFunctional(f,spaces...,measure)
end

abstract type ParamFEQuantity{FS,S,TT} end

abstract type ParamFEFunctional{FS,S,TT} <: ParamFEQuantity{FS,S,TT} end

mutable struct ParamLinFEFunctional{FS,S} <: ParamFEFunctional{FS,S,nothing}
  param_functional::ParamFunctional{FS,S}
  fe_functional::LinFEFunctional
end

mutable struct ParamBilinFEFunctional{FS,S,TT} <: ParamFEFunctional{FS,S,TT}
  param_functional::ParamFunctional{FS,S}
  fe_functional::BilinFEFunctional{TT}
end

function ParamFEFunctional(
  Fμ::ParamFunctional{FS,S},
  Fv::LinFEFunctional) where {FS,S}

  ParamLinFEFunctional{FS,S}(Fμ,Fv)
end

function ParamFEFunctional(
  Fμ::ParamFunctional{FS,S},
  Fuv::BilinFEFunctional{TT}) where {FS,S,TT}

  ParamBilinFEFunctional{FS,S,TT}(Fμ,Fuv)
end

function get_fd_dofs(F::ParamBilinFEFunctional)
  trial = F.fe_functional.trial
  fdofs = free_dofs_on_full_trian(trial)
  ddofs = trial.ddofs_on_full_trian
  fdofs,ddofs
end

function _compose_functionals(
  Fμ::ParamFunctional{FS,true},
  Fuv::FEFunctional{1,TT}) where {FS,TT}

  form(μ::Vector{Float},v) = ∫(Fuv.f(Fμ.f(μ),v))Fuv.measure
  form(μ::Vector{Float}) = v -> form(μ,v)
  form
end

function _compose_functionals(
  Fμ::ParamFunctional{FS,true},
  Fuv::FEFunctional{2,TT}) where {FS,TT}

  form(μ::Vector{Float},u,v) = ∫(Fuv.f(Fμ.f(μ),u,v))Fuv.measure
  form(μ::Vector{Float}) = (u,v) -> form(μ,u,v)
  form
end

function _compose_functionals(
  Fμ::ParamFunctional{FS,false},
  Fuv::FEFunctional{1,TT}) where {FS,TT}

  form(μ::Vector{Float},t,v) = ∫(Fuv.f(Fμ.f(μ,t),t,v))Fuv.measure
  form(μ::Vector{Float},t) = v -> form(μ,t,v)
  form(μ::Vector{Float}) = t -> form(μ,t)
  form
end

function _compose_functionals(
  Fμ::ParamFunctional{FS,false},
  Fuv::FEFunctional{2,TT}) where {FS,TT}

  form(μ::Vector{Float},t,u,v) = ∫(Fuv.f(Fμ.f(μ,t),t,u,v))Fuv.measure
  form(μ::Vector{Float},t) = (u,v) -> form(μ,t,u,v)
  form(μ::Vector{Float}) = t -> form(μ,t)
  form
end

function realization(
  Fuvμ::ParamFEFunctional{FS,S,TT}) where {FS,S,TT}

  Fuv, Fμ = Fuvμ.fe_functional, Fuvμ.param_functional
  param_form = _compose_functionals(Fμ,Fuv)
  μ, Fμ_μ = realization(Fμ.param_functional)
  μ, param_form(Fμ_μ)
end

abstract type ParamFEArray{FS,S,TT,N} <: ParamFEQuantity{FS,S,TT} end

mutable struct ParamFEVector{FS,S} <: ParamFEArray{FS,S,nothing,1}
  id::String
  param_fe_functional::ParamLinFEFunctional{FS,S}
  array::Function
end

mutable struct ParamFEMatrix{FS,S,TT} <: ParamFEArray{FS,S,TT,2}
  id::String
  param_fe_functional::ParamBilinFEFunctional{FS,S,TT}
  array::Function
end

function Base.getproperty(fe_array::ParamFEArray,sym::Symbol)
  if sym ∈ (:param_functional,:fe_functional)
    getfield(fe_array.param_fe_functional,sym)
  else
    getfield(fe_array, sym)
  end
end

function Base.setproperty!(fe_array::ParamFEArray,sym::Symbol,x)
  if sym ∈ (:param_functional,:fe_functional)
    setfield!(fe_array.param_fe_functional,sym,x)
  else
    setfield!(fe_array,sym,x)
  end
end

function ParamFEArray(
  id::String,
  Fvμ::ParamLinFEFunctional{FS,S}) where {FS,S}

  ParamFEVector{FS,S}(id,Fvμ,_assemble_array(Fvμ))
end

function ParamFEArray(
  id::String,
  Fuvμ::ParamBilinFEFunctional{FS,S,TT}) where {FS,S,TT}

  ParamFEMatrix{FS,S,TT}(id,Fuvμ,_assemble_array(Fuvμ))
end

function ParamFEArray(
  id::String,
  Fμ::ParamFunctional{FS,S},
  Fuv::FEFunctional{N,TT}) where {FS,N,S,TT}

  Fuvμ = ParamFEFunctional(Fμ,Fuv)
  ParamFEArray(id,Fuvμ)
end

function _assemble_array(Fvμ::ParamLinFEFunctional)
  V = get_test(Fvμ)
  lin_form = _compose_functionals(Fvμ.param_functional,Fvμ.fe_functional)
  μ -> assemble_vector(lin_form(μ),V)
end

function _assemble_array(Fuvμ::ParamBilinFEFunctional{FS,S,TT}) where {FS,S,TT}

  U_no_bc,V_no_bc = get_trial_no_bc(Fuvμ),get_test_no_bc(Fuvμ)
  fdofs,ddofs = get_fd_dofs(Fuvμ)

  bilin_form = _compose_functionals(Fuvμ.param_functional,Fuvμ.fe_functional)
  mat_all_dofs(μ) = assemble_matrix(bilin_form(μ),U_no_bc,V_no_bc)
  mat_free_dofs(μ) = mat_all_dofs(μ)[fdofs,fdofs]

  if !has_dirichlet_bc(Fuvμ)
    μ -> [mat_free_dofs(μ)]
  else
    μ -> [mat_free_dofs(μ),_assemble_lift(mat_all_dofs,get_trial(Fuvμ),fdofs,ddofs)(μ)]
  end
end

function _assemble_lift(
  mat_all_dofs::Function,
  trial::MyTrial,
  fdofs::Vector{Int},
  ddofs::Vector{Int})

  g(μ) = trial.trial(μ).dirichlet_values
  μ -> mat_all_dofs(μ)[fdofs,ddofs]*g(μ)
end

get_trial(q::ParamFEQuantity) = q.fe_functional.trial.trial
get_trial_no_bc(q::ParamFEQuantity) = q.fe_functional.trial.trial_no_bc
get_trials(q::ParamFEQuantity) = get_trial(q), get_trial_no_bc(q)
get_test(q::ParamFEQuantity) = q.fe_functional.test.test
get_test_no_bc(q::ParamFEQuantity) = q.fe_functional.test.test_no_bc
get_tests(q::ParamFEQuantity) = get_test(q), get_test_no_bc(q)
get_measure(q::ParamFEQuantity) = q.fe_functional.measure
Gridap.get_triangulation(q::ParamFEQuantity) = get_triangulation(get_test(q))

isaffine(::ParamFEQuantity{FS,S,TT}) where {FS,S,TT} = (FS == Affine())
islinear(::ParamFEQuantity{FS,S,TT}) where {FS,S,TT} = !(FS == Nonlinear())
issteady(::ParamFEQuantity{FS,S,TT}) where {FS,S,TT} = S
has_dirichlet_bc(::ParamFEQuantity{FS,S,TT}) where {FS,S,TT} =
  (TT ∈ (ParamTrialFESpace,ParamTransientTrialFESpace))

mutable struct ParamFEProblem{D,I,S}
  param_space::ParamSpace
  param_fe_functional::Vector{<:ParamFEFunctional}
  param_fe_array::Vector{<:ParamFEArray}

  function ParamFEProblem(
    param_space::ParamSpace,
    param_fe_functional::Vector{<:ParamFEFunctional},
    param_fe_array::Vector{<:ParamFEArray};
    I=true,S=false)

    D = num_cell_dims(get_triangulation(first(param_fe_functional)))
    new{D,I,S}(param_space,param_fe_functional,param_fe_array)
  end
end

function ParamFEProblem(
  id::String,
  FS::FunctionalStyle,
  param_fun::Function,
  fe_fun::Function,
  param_space::ParamSpace,
  fe_spaces::Tuple{MyTests},
  measure::Measure;
  S=false)

  param_functional = ParamFunctional(param_space,param_fun;FS,S)
  fe_functional = FEFunctional(fe_fun,fe_spaces,measure)
  param_fe_functional = ParamFEFunctional(param_functional,fe_functional)
  param_fe_array = ParamFEArray(id,param_fe_functional)

  param_fe_functional,param_fe_array
end

function ParamFEProblem(
  id::String,
  FS::FunctionalStyle,
  param_fun::Function,
  fe_fun::Function,
  param_space::ParamSpace,
  fe_spaces::Tuple{MyTrials,MyTests},
  measure::Measure;
  S=false)

  param_functional = ParamFunctional(param_space,param_fun;FS,S)
  fe_functional = FEFunctional(fe_fun,fe_spaces,measure)
  param_fe_functional = ParamFEFunctional(param_functional,fe_functional)
  param_fe_array = ParamFEArray(id,param_fe_functional)

  param_fe_functional,param_fe_array
end

function ParamFEProblem(
  id::Vector{<:String},
  FS::Vector{<:FunctionalStyle},
  param_fun::Vector{<:Function},
  fe_fun::Vector{<:Function},
  param_space::ParamSpace,
  fe_spaces::Vector{<:Tuple},
  measure::Vector{<:Measure};
  I=true,S=false)

  param_quantities =
    (Broadcasting((i,a,pf,ff,fs,m)->
    ParamFEProblem(i,a,pf,ff,param_space,fs,m;S))(id,FS,param_fun,fe_fun,fe_spaces,measure))
  param_fe_functional,param_fe_array = first.(param_quantities),last.(param_quantities)

  ParamFEProblem(param_space,param_fe_functional,param_fe_array;I,S)
end

function ParamFEProblem(
  dict::Dict{Symbol,Vector};
  I=true,S=false)

  @unpack id,FS,param_fun,fe_fun,param_space,fe_spaces,measure = dict
  ParamFEProblem(id,FS,param_fun,fe_fun,param_space...,fe_spaces,measure;I,S)
end
