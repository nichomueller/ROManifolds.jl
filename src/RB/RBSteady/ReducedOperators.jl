"""
    reduced_operator(solver::RBSolver,feop::ParamOperator,args...;kwargs...) -> RBOperator
    reduced_operator(solver::RBSolver,feop::TransientParamOperator,args...;kwargs...) -> TransientRBOperator

Computes a RB operator from the FE operator `feop`
"""
function reduced_operator(
  dir::String,
  solver::RBSolver,
  feop::ParamOperator,
  args...;
  kwargs...)

  fesnaps,festats = solution_snapshots(solver,feop,args...;kwargs...)
  rbop = reduced_operator(solver,feop,fesnaps)
  save(dir,fesnaps)
  save(dir,rbop)
  rbop
end

function reduced_operator(
  solver::RBSolver,
  feop::ParamOperator,
  s::AbstractSnapshots)

  red_trial,red_test = reduced_spaces(solver,feop,s)
  reduced_operator(solver,feop,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  feop::ParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  red_lhs,red_rhs = reduced_weak_form(solver,feop,red_trial,red_test,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = get_domains(red_lhs)
  feop′ = change_domains(feop,trians_rhs,trians_lhs)
  GenericRBOperator(feop′,red_trial,red_test,red_lhs,red_rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  red_op_lin = reduced_operator(solver,get_linear_operator(op),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),red_trial,red_test,s)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

"""
    abstract type RBOperator{O} <: ParamOperator{O,SplitDomains} end

Type representing reduced algebraic operators used within a reduced order modelling
framework in steady applications. A RBOperator should contain the following information:

- a reduced test and trial space, computed according to [`reduced_spaces`](@ref)
- a hyper-reduced residual and jacobian, computed according to [`reduced_weak_form`](@ref)

Subtypes:

- [`GenericRBOperator`](@ref)
- [`LinearNonlinearRBOperator`](@ref)
"""
abstract type RBOperator{O} <: ParamOperator{O,SplitDomains} end

"""
    struct GenericRBOperator{O,A} <: RBOperator{O}
      op::ParamOperator{O}
      trial::RBSpace
      test::RBSpace
      lhs::A
      rhs::AffineContribution
    end

Fields:

- `op`: underlying high dimensional FE operator
- `trial`: reduced trial space
- `test`: reduced trial space
- `lhs`: hyper-reduced left hand side
- `rhs`: hyper-reduced right hand side
"""
struct GenericRBOperator{O,A} <: RBOperator{O}
  op::ParamOperator{O}
  trial::RBSpace
  test::RBSpace
  lhs::A
  rhs::AffineContribution
end

FESpaces.get_trial(op::GenericRBOperator) = op.trial
FESpaces.get_test(op::GenericRBOperator) = op.test
ParamSteady.get_fe_operator(op::GenericRBOperator) = op.op

function Algebra.allocate_residual(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  allocate_hypred_cache(op.rhs,r)
end

function Algebra.allocate_jacobian(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  allocate_hypred_cache(op.lhs,r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)

  trian_res = get_domains_res(op.op)
  res = get_res(op.op)
  dc = res(r,uh,v)

  map(trian_res) do strian
    b_strian = b.fecache[strian]
    rhs_strian = op.rhs[strian]
    vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian)
    assemble_hr_vector_add!(b_strian,vecdata...)
  end

  inv_project!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op.op)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)

  trian_jac = get_domains_jac(op.op)
  jac = get_jac(op.op)
  dc = jac(r,uh,du,v)

  map(trian_jac) do strian
    A_strian = A.fecache[strian]
    lhs_strian = op.lhs[strian]
    matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian)
    assemble_hr_matrix_add!(A_strian,matdata...)
  end

  inv_project!(A,op.lhs)
end

const ParamNonlinearRBOperator = GenericParamNonlinearOperator{<:RBOperator}

"""
    struct LinearNonlinearRBOperator{A} <: RBOperator{LinearNonlinearParamEq}
      op_linear::GenericRBOperator{LinearParamEq,A}
      op_nonlinear::GenericRBOperator{NonlinearParamEq,A}
    end

Extends the concept of [`GenericRBOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications
"""
struct LinearNonlinearRBOperator{A} <: RBOperator{LinearNonlinearParamEq}
  op_linear::GenericRBOperator{LinearParamEq,A}
  op_nonlinear::GenericRBOperator{NonlinearParamEq,A}
end

ParamAlgebra.get_linear_operator(op::LinearNonlinearRBOperator) = op.op_linear
ParamAlgebra.get_nonlinear_operator(op::LinearNonlinearRBOperator) = op.op_nonlinear
FESpaces.get_trial(op::LinearNonlinearRBOperator) = get_trial(get_nonlinear_operator(op))
FESpaces.get_test(op::LinearNonlinearRBOperator) = get_test(get_nonlinear_operator(op))

const LinNonlinRBOperator = LinNonlinParamOperator{<:ParamNonlinearRBOperator,<:ParamNonlinearRBOperator}

# function ParamAlgebra.allocate_systemcache(nlop::LinNonlinRBOperator,x::AbstractVector)
#   cache_linear = get_linear_systemcache(nlop)
#   feop = get_fe_operator(nlop.op_nonlinear.op)
#   trian_jac = get_domains_jac(feop)
#   trian_res = get_domains_res(feop)
#   A_nlin = change_domains(cache_linear.A,trian_jac)
#   b_nlin = change_domains(cache_linear.b,trian_res)
#   SystemCache(A_nlin,b_nlin)
# end
