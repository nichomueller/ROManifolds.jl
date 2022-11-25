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

struct Nonaffine <: OperatorType end
abstract type ParamVarOperator{OT,S,TT} end

struct ParamLinOperator{OT,S} <: ParamVarOperator{OT,S,nothing}
  a::Function
  afe::Function
  A::Function
  pparam::ParamSpace
  tests::MyTests
end

struct ParamBilinOperator{OT,S,TT} <: ParamVarOperator{OT,S,TT}
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
  OT=Nonaffine(),S=false)

  A(μ) = assemble_vector(afe(μ),tests.test)
  ParamLinOperator{OT,S}(a,afe,A,pparam,tests)
end

function ParamVarOperator(
  a::Function,
  afe::Function,
  pparam::ParamSpace,
  trials::MyTrials{TT},
  tests::MyTests;
  OT=Nonaffine(),S=false) where TT

  A = assemble_matrix_and_lifting(afe,trials,tests)
  ParamBilinOperator{OT,S,TT}(a,afe,A,pparam,trials,tests)
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
