abstract type SamplingStyle end
struct UniformSampling <: SamplingStyle end
struct NormalSampling <: SamplingStyle end

#= abstract type ParamSpace end

struct MyParamSpace <: ParamSpace
  ρ::Float
  ν::Function
  f::Function

  u::FEFunction
end

struct ParamSample{P<:MyParamSpace}
  cache
end

function realization(P::MyParamSpace)
  uvec = rand(get_free_dof_values(P.u))
  cache = uvec
  ParamSample{P}(cache)
end =#

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

mutable struct ParamFunctional{S}
  param_space::ParamSpace
  f::Function

  function ParamFunctional(
    P::ParamSpace,
    f::Function;
    S=false)
    new{S}(P,f)
  end
end

function realization(Fμ::ParamFunctional{S}) where S
  μ = realization(Fμ.param_space)
  μ, Fμ.f(μ)
end

function realization(
  P::ParamSpace,
  f::Function;
  S=false)

  Fμ = ParamFunctional(P,f;S)
  realization(Fμ)
end

abstract type MySpaces end

function dirichlet_dofs_on_full_trian(space,space_no_bc)

  cell_dof_ids = get_cell_dof_ids(space)
  cell_dof_ids_no_bc = get_cell_dof_ids(space_no_bc)

  dirichlet_dofs = zeros(Int,space.ndirichlet)
  for cell = eachindex(cell_dof_ids)
    for (ids,ids_no_bc) in zip(cell_dof_ids[cell],cell_dof_ids_no_bc[cell])
      if ids<0
        dirichlet_dofs[abs(ids)]=ids_no_bc
      end
    end
  end

  dirichlet_dofs
end

struct MyTests <: MySpaces
  test::UnconstrainedFESpace
  test_no_bc::UnconstrainedFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTests(
    test::UnconstrainedFESpace,
    test_no_bc::UnconstrainedFESpace)

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(test,test_no_bc)
    new(test,test_no_bc,ddofs_on_full_trian)
  end
end

function MyTests(model,reffe;kwargs...)
  test = TestFESpace(model,reffe;kwargs...)
  test_no_bc = FESpace(model,reffe)
  MyTests(test,test_no_bc)
end

function MyTrial(test::UnconstrainedFESpace)
  HomogeneousTrialFESpace(test)
end

function MyTrial(
  test::UnconstrainedFESpace,
  Gμ::ParamFunctional{true})
  ParamTrialFESpace(test,Gμ.f)
end

function MyTrial(
  test::UnconstrainedFESpace,
  Gμ::ParamFunctional{false})
  ParamTransientTrialFESpace(test,Gμ.f)
end

struct MyTrials{TT} <: MySpaces
  trial::TT
  trial_no_bc::UnconstrainedFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTrials(
    trial::TT,
    trial_no_bc::UnconstrainedFESpace) where TT

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(trial.space,trial_no_bc)
    new{TT}(trial,trial_no_bc,ddofs_on_full_trian)
  end
end

function MyTrials(tests::MyTests,args...)
  trial = MyTrial(tests.test,args...)
  trial_no_bc = TrialFESpace(tests.test_no_bc)
  MyTrials(trial,trial_no_bc)
end

function ParamMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamMultiFieldFESpace([first(spaces).trial,last(spaces).trial])
end

function ParamMultiFieldFESpace(spaces::Vector{MyTests})
  ParamMultiFieldFESpace([first(spaces).test,last(spaces).test])
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamTransientMultiFieldFESpace([first(spaces).trial,last(spaces).trial])
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTests})
  ParamTransientMultiFieldFESpace([first(spaces).test,last(spaces).test])
end

function free_dofs_on_full_trian(tests::MyTests)
  nfree_on_full_trian = tests.test_no_bc.nfree
  setdiff(collect(1:nfree_on_full_trian),tests.ddofs_on_full_trian)
end

function free_dofs_on_full_trian(trials::MyTrials)
  nfree_on_full_trian = trials.trial_no_bc.nfree
  setdiff(collect(1:nfree_on_full_trian),trials.ddofs_on_full_trian)
end

function get_fd_dofs(tests::MyTests,trials::MyTrials)
  fdofs_test = free_dofs_on_full_trian(tests)
  fdofs_trial = free_dofs_on_full_trian(trials)
  ddofs = trials.ddofs_on_full_trian
  (fdofs_test,fdofs_trial),ddofs
end

abstract type ParamVarOperator{FS,S,TT} end

struct ParamLinOperator{FS,S} <: ParamVarOperator{FS,S,nothing}
  a::Function
  afe::Function
  A::Function
  pparam::ParamSpace
  tests::MyTests
end

struct ParamBilinOperator{FS,S,TT} <: ParamVarOperator{FS,S,TT}
  a::Function
  afe::Function
  A::Vector{<:Function}
  pparam::ParamSpace
  trials::MyTrials{TT}
  tests::MyTests
end

function ParamVarOperator(
  a::Function,
  afe::Function,
  pparam::ParamSpace,
  tests::MyTests;
  FS=Nonaffine(),S=false)

  A(μ) = assemble_vector(afe(μ),tests.test)
  ParamLinOperator{FS,S}(a,afe,A,pparam,tests)
end

function ParamVarOperator(
  a::Function,
  afe::Function,
  pparam::ParamSpace,
  trials::MyTrials{TT},
  tests::MyTests;
  FS=Nonaffine(),S=false) where TT

  A = assemble_matrix_and_lifting(afe,trials,tests)
  ParamBilinOperator{FS,S,TT}(a,afe,A,pparam,trials,tests)
end

function assemble_matrix_and_lifting(
  afe::Function,
  trials::MyTrials{TT},
  tests::MyTests) where TT

  U,U_no_bc,V_no_bc = trials.trial,trials.trial_no_bc,tests.test_no_bc
  fdofs,ddofs = get_fd_dofs(tests,trials)
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ),U_no_bc,V_no_bc)
  A_bc(μ) = A_no_bc(μ)[fdofs_test,fdofs_trial]
  dir(μ) = U(μ).dirichlet_values

  [μ -> A_bc(μ),μ -> A_no_bc(μ)[fdofs_test,ddofs]*dir(μ)]
end

function assemble_matrix_and_lifting(
  afe::Function,
  trials::MyTrials{TrialFESpace},
  tests::MyTests)

  [μ -> assemble_matrix(afe(μ),trials.trial,tests.test)]
end

function Gridap.FEFunction(
  spaces::Tuple{ParamSpace,MyTrials},
  values::Tuple)

  _,trials = spaces
  μ,free_values = values
  FEFunction(trials.trial(μ),free_values)
end

function Gridap.FEFunction(
  spaces::Tuple{ParamSpace,MyTests},
  values::Tuple)

  _,tests = spaces
  μ,free_values = values
  FEFunction(tests.test(μ),free_values)
end
