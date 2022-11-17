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

function compose_functionals(
  Fμ::ParamFunctional{FS,true},
  Fuv::LinFEFunctional) where FS

  form(μ::Vector{Float},v) = ∫(Fuv.f(Fμ.f(μ),v))Fuv.measure
  form(μ::Vector{Float}) = v -> form(μ,v)
  form
end

function compose_functionals(
  Fμ::ParamFunctional{FS,true},
  Fuv::BilinFEFunctional{TT}) where {FS,TT}

  form(μ::Vector{Float},u,v) = ∫(Fuv.f(Fμ.f(μ),u,v))Fuv.measure
  form(μ::Vector{Float}) = (u,v) -> form(μ,u,v)
  form
end

function compose_functionals(
  Fμ::ParamFunctional{FS,false},
  Fuv::LinFEFunctional) where FS

  form(μ::Vector{Float},t,v) = ∫(Fuv.f(Fμ.f(μ,t),t,v))Fuv.measure
  form(μ::Vector{Float},t) = v -> form(μ,t,v)
  form(μ::Vector{Float}) = t -> form(μ,t)
  form
end

function compose_functionals(
  Fμ::ParamFunctional{FS,false},
  Fuv::BilinFEFunctional{TT}) where {FS,TT}

  form(μ::Vector{Float},t,u,v) = ∫(Fuv.f(Fμ.f(μ,t),t,u,v))Fuv.measure
  form(μ::Vector{Float},t) = (u,v) -> form(μ,t,u,v)
  form(μ::Vector{Float}) = t -> form(μ,t)
  form
end

function compose_functionals(Fuvμ::ParamFEFunctional)
  compose_functionals(Fuvμ.param_functional,Fuvμ.fe_functional)
end

#= function compose_nonlinear_functionals(
  Fuvμ::ParamBilinFEFunctional{Nonlinear,S,TT}) where {S,TT}

  nonlin_form = compose_functionals(Fuvμ)
  (u,v) -> nonlin_form(u,u,v)
end =#

function realization(
  Fuvμ::ParamFEFunctional{FS,S,TT}) where {FS,S,TT}

  param_form = compose_functionals(Fuvμ)
  μ, Fμ_μ = realization(Fuvμ.param_functional)
  μ, param_form(Fμ_μ)
end

abstract type ParamFEArray{FS,S,TT,N} <: ParamFEQuantity{FS,S,TT} end

mutable struct ParamFEVector{FS,S} <: ParamFEArray{FS,S,nothing,1}
  id::Symbol
  param_fe_functional::ParamLinFEFunctional{FS,S}
  array::Function
end

mutable struct ParamFEMatrix{FS,S,TT} <: ParamFEArray{FS,S,TT,2}
  id::Symbol
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
  id::Symbol,
  Fvμ::ParamLinFEFunctional{FS,S}) where {FS,S}

  ParamFEVector{FS,S}(id,Fvμ,assemble_array(Fvμ))
end

function ParamFEArray(
  id::Symbol,
  Fuvμ::ParamBilinFEFunctional{FS,S,TT}) where {FS,S,TT}

  ParamFEMatrix{FS,S,TT}(id,Fuvμ,assemble_array(Fuvμ))
end

function ParamFEArray(
  id::Symbol,
  Fμ::ParamFunctional{FS,S},
  Fuv::FEFunctional{N,TT}) where {FS,N,S,TT}

  Fuvμ = ParamFEFunctional(Fμ,Fuv)
  ParamFEArray(id,Fuvμ)
end

function assemble_array(Fvμ::ParamLinFEFunctional)
  V = get_test(Fvμ)
  lin_form = compose_functionals(Fvμ.param_functional,Fvμ.fe_functional)
  μ -> assemble_vector(lin_form(μ),V)
end

function assemble_array(Fuvμ::ParamBilinFEFunctional{FS,S,TT}) where {FS,S,TT}

  U_no_bc,V_no_bc = get_trial_no_bc(Fuvμ),get_test_no_bc(Fuvμ)
  fdofs,ddofs = get_fd_dofs(Fuvμ)

  bilin_form = compose_functionals(Fuvμ.param_functional,Fuvμ.fe_functional)
  mat_all_dofs(μ) = assemble_matrix(bilin_form(μ),U_no_bc,V_no_bc)
  mat_free_dofs(μ) = mat_all_dofs(μ)[fdofs,fdofs]

  if !has_dirichlet_bc(Fuvμ)
    μ -> [mat_free_dofs(μ)]
  else
    μ -> [mat_free_dofs(μ),assemble_lift(mat_all_dofs,get_trial(Fuvμ),fdofs,ddofs)(μ)]
  end
end

function assemble_lift(
  mat_all_dofs::Function,
  trial::MyTrial,
  fdofs::Vector{Int},
  ddofs::Vector{Int})

  g(μ) = trial.trial(μ).dirichlet_values
  μ -> mat_all_dofs(μ)[fdofs,ddofs]*g(μ)
end

get_trial(q::ParamBilinFEFunctional) = q.fe_functional.trial.trial
get_trial_no_bc(q::ParamBilinFEFunctional) = q.fe_functional.trial.trial_no_bc
get_trial(::ParamLinFEFunctional) = nothing
get_trial_no_bc(::ParamLinFEFunctional) = nothing
get_trial(q::ParamFEQuantity) = get_trial(q.fe_functional)
get_trial_no_bc(q::ParamFEQuantity) = get_trial_no_bc(q.fe_functional)
get_trial(q::ParamFEQuantity) = get_trial(q), get_trial_no_bc(q)
get_test(q::ParamFEQuantity) = q.fe_functional.test.test
get_test_no_bc(q::ParamFEQuantity) = q.fe_functional.test.test_no_bc
get_tests(q::ParamFEQuantity) = get_test(q), get_test_no_bc(q)
get_param_space(q::ParamFEQuantity) = q.param_functional.param_space
get_measure(q::ParamFEQuantity) = q.fe_functional.measure
get_reffe(q::ParamFEQuantity) = first(get_measure(q).quad.trian.model.grid.reffes)
Gridap.get_triangulation(q::ParamFEQuantity) = get_triangulation(get_test(q))
get_model(q::ParamFEQuantity) = get_triangulation(q).model
is_bilinear(q::ParamFEQuantity) = typeof(q.fe_functional) <: BilinFEFunctional
get_id(q::ParamFEArray) = q.id

isaffine(::ParamFEQuantity{FS,S,TT}) where {FS,S,TT} = (FS == Affine())
islinear(::ParamFEQuantity{FS,S,TT}) where {FS,S,TT} = !(FS == Nonlinear())
issteady(::ParamFEQuantity{FS,S,TT}) where {FS,S,TT} = S
has_dirichlet_bc(::ParamFEQuantity{FS,S,TT}) where {FS,S,TT} =
  (TT ∈ (ParamTrialFESpace,ParamTransientTrialFESpace))

get_affinity(::ParamFEQuantity{FS,S,TT}) where {FS,S,TT} = FS
get_affinity(::Vector{<:FunctionalStyle}) = Nonaffine()
get_affinity(::Vector{Affine}) = Affine()
get_affinity(::Vector{Nonlinear}) = Nonlinear()
get_affinity(q::Vector{<:ParamFEQuantity}) =
  get_affinity(Broadcasting(get_affinity)(q))

function get_physical_quadrature_points(q::ParamFEQuantity)
  meas, trian = get_measure(q), get_triangulation(q)

  map_element = get_cell_map(trian)
  quad_element = get_data(get_cell_points(meas.quad))
  map(Gridap.evaluate,map_element,quad_element)
end

function get_fespace_on_quadrature_points(q::ParamFEQuantity)
  model = get_model(q)
  reffe = get_reffe(q)
  order = first(get_orders(reffe))

  reffe_quad = Gridap.ReferenceFE(lagrangian_quad,Float,order)
  TestFESpace(model,reffe_quad,conformity=:L2)
end

mutable struct ParamFEProblem{A,D,I,S}
  param_fe_functional::Vector{<:ParamFEFunctional}
  param_fe_array::Vector{<:ParamFEArray}

  function ParamFEProblem(
    param_fe_functional::Vector{<:ParamFEFunctional},
    param_fe_array::Vector{<:ParamFEArray};
    I=true,S=false)

    A = get_affinity(param_fe_functional)
    D = num_cell_dims(get_triangulation(first(param_fe_functional)))
    new{A,D,I,S}(param_fe_functional,param_fe_array)
  end
end

function ParamFEProblem(
  id::Symbol,
  FS::FunctionalStyle,
  param_fun::Function,
  fe_fun::Function,
  param_space::ParamSpace,
  fe_spaces::Tuple,
  measure::Measure;
  S=false)

  param_functional = ParamFunctional(param_space,param_fun;FS,S)
  fe_functional = FEFunctional(fe_fun,fe_spaces,measure)
  param_fe_functional = ParamFEFunctional(param_functional,fe_functional)
  param_fe_array = ParamFEArray(id,param_fe_functional)

  param_fe_functional,param_fe_array
end

function ParamFEProblem(
  id::Vector{<:Symbol},
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

  ParamFEProblem(param_fe_functional,param_fe_array;I,S)
end

function ParamFEProblem(
  dict::Dict{Symbol,Vector};
  I=true,S=false)

  @unpack id,FS,param_fun,fe_fun,param_space,fe_spaces,measure = dict
  ParamFEProblem(id,FS,param_fun,fe_fun,param_space...,fe_spaces,measure;I,S)
end

function get_all_trial(p::ParamFEProblem)
  my_trial = Broadcasting(get_trial)(p.param_fe_functional)
  Broadcasting(mt->getproperty(mt,:trial))(my_trial)
end

function get_trial(p::ParamFEProblem)
  my_trial_tmp = unique(Broadcasting(get_trial)(p.param_fe_functional))
  my_trial = my_trial_tmp[.!isnothing.(my_trial_tmp)]
  Broadcasting(mt->getproperty(mt,:trial))(my_trial)
end

get_trial_no_bc(p::ParamFEProblem) = unique(Broadcasting(get_trial_no_bc)(p.param_fe_functional))
get_all_test(p::ParamFEProblem) = Broadcasting(get_test)(p.param_fe_functional)
get_test(p::ParamFEProblem) = unique(Broadcasting(get_test)(p.param_fe_functional))
get_test_no_bc(p::ParamFEProblem) = unique(Broadcasting(get_test_no_bc)(p.param_fe_functional))
get_tests(p::ParamFEProblem) = unique(Broadcasting(get_tests)(p.param_fe_functional))
get_param_space(p::ParamFEProblem) = get_param_space(first(p.param_fe_functional))
get_measure(p::ParamFEProblem) = unique(Broadcasting(get_measure)(p.param_fe_functional))
is_bilinear(p::ParamFEProblem) = Broadcasting(is_bilinear)(p.param_fe_functional)
get_id(p::ParamFEProblem) = Broadcasting(get_id)(p.param_fe_array)

isaffine(p::ParamFEProblem) = all(Broadcasting(isaffine)(p.param_fe_functional))
islinear(p::ParamFEProblem) = all(Broadcasting(islinear)(p.param_fe_functional))
issteady(p::ParamFEProblem) = all(Broadcasting(issteady)(p.param_fe_functional))
has_dirichlet_bc(p::ParamFEProblem) = Broadcasting(has_dirichlet_bc)(p.param_fe_functional)

function Gridap.FEFunction(
  spaces::Tuple{ParamSpace,MyTrial},
  values::NTuple{2,Vector{Float}})

  _,trial = spaces
  μ,free_values = values
  FEFunction(trial.trial(μ),free_values)
end
