struct ParamLiftingOperator{OT,TT} <: ParamVarOperator{OT,TT}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  trials::MyTrials{TT}
  tests::MyTests
end

function ParamLiftingOperator(op::ParamBilinOperator{OT,TT}) where {OT,TT}
  id,a,afe,pspace,tests,trials = get_id(op),get_param_fun(op),get_fe_fun(op),get_pspace(op),get_tests(op),get_trials(op)
  ParamLinOperator{OT,TT}(id,a,afe,pspace,trials,tests)
end

get_id(op::ParamVarOperator) = op.id
get_param_function(op::ParamVarOperator) = op.a
get_fe_function(op::ParamVarOperator) = op.afe
Gridap.ODEs.TransientFETools.get_test(op::ParamVarOperator) = op.tests.test
Gridap.ODEs.TransientFETools.get_trial(op::ParamBilinOperator) = op.trials.trial
get_tests(op::ParamVarOperator) = op.tests
get_trials(op::ParamBilinOperator) = op.trials
get_test_no_bc(op::ParamVarOperator) = op.tests.test_no_bc
get_trial_no_bc(op::ParamBilinOperator) = op.trials.trial_no_bc
get_pspace(op::ParamVarOperator) = op.pspace

function Gridap.FESpaces.assemble_vector(op::ParamLiftingOperator)
  afe = get_fe_function(op)
  trial = get_trial(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ),trial_no_bc,test_no_bc)
  dir(μ) = trial(μ).dirichlet_values
  lift(μ) = A_no_bc(μ)[fdofs_test,ddofs]*dir(μ)

  lift
end
