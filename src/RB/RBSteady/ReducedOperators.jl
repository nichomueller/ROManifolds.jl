"""
    reduced_operator(solver::RBSolver,feop::ParamFEOperator,args...;kwargs...) -> RBOperator
    reduced_operator(solver::RBSolver,feop::TransientParamFEOperator,args...;kwargs...) -> TransientRBOperator

Computes a RB operator from the FE operator `feop`
"""
function reduced_operator(
  dir::String,
  solver::RBSolver,
  feop::ParamFEOperator,
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
  feop::ParamFEOperator,
  s::AbstractSnapshots)

  red_trial,red_test = reduced_spaces(solver,feop,s)
  op = get_algebraic_operator(feop)
  reduced_operator(solver,op,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  red_lhs,red_rhs = reduced_weak_form(solver,op,red_trial,red_test,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = get_domains(red_lhs)
  op′ = change_domains(op,trians_rhs,trians_lhs)
  GenericRBOperator(op′,red_trial,red_test,red_lhs,red_rhs)
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
    struct GenericRBOperator{O} <: RBOperator{O}
      op::ParamOperator{O}
      trial::RBSpace
      test::RBSpace
      lhs::AffineContribution
      rhs::AffineContribution
    end

Fields:

- `op`: underlying high dimensional FE operator
- `trial`: reduced trial space
- `test`: reduced trial space
- `lhs`: hyper-reduced left hand side
- `rhs`: hyper-reduced right hand side
"""
struct GenericRBOperator{A,O} <: RBOperator{O}
  op::ParamOperator{O}
  trial::RBSpace
  test::RBSpace
  lhs::A
  rhs::AffineContribution
end

FESpaces.get_trial(op::GenericRBOperator) = op.trial
FESpaces.get_test(op::GenericRBOperator) = op.test
get_fe_trial(op::GenericRBOperator) = get_trial(op.op)
get_fe_test(op::GenericRBOperator) = get_test(op.op)

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

  hr_residual!(b,op,r,u,paramcache)
  inv_project!(b,op.rhs,feb)
end

# function Algebra.residual!(
#   b::HRParamArray,
#   op::GenericRBOperator,
#   r::Realization,
#   u::RBParamVector,
#   paramcache)

#   inv_project!(u.fe_data,paramcache.trial,u.data)
#   residual!(b,op,r,u.fe_data,paramcache)
# end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  hr_jacobian!(A,op,r,u,paramcache)
  inv_project!(A,op.lhs)
end

# function Algebra.jacobian!(
#   A::HRParamArray,
#   op::GenericRBOperator,
#   r::Realization,
#   u::RBParamVector,
#   paramcache)

#   inv_project!(u.fe_data,paramcache.trial,u.data)
#   jacobian!(A,op,r,u.fe_data,rbcache)
# end

"""
    struct LinearNonlinearRBOperator <: RBOperator{LinearNonlinearParamEq}
      op_linear::GenericRBOperator{LinearParamEq}
      op_nonlinear::GenericRBOperator{NonlinearParamEq}
    end

Extends the concept of [`GenericRBOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications
"""
struct LinearNonlinearRBOperator <: RBOperator{LinearNonlinearParamEq}
  op_linear::GenericRBOperator{LinearParamEq}
  op_nonlinear::GenericRBOperator{NonlinearParamEq}
end

ParamAlgebra.get_linear_operator(op::LinearNonlinearRBOperator) = op.op_linear
ParamAlgebra.get_nonlinear_operator(op::LinearNonlinearRBOperator) = op.op_nonlinear

# cache utils

# selects the entries of the snapshots relevant to the reduced integration domain
# in `a`
function select_at_indices(s::AbstractArray,a::HyperReduction)
  s[get_integration_domain(a)]
end

function Arrays.return_cache(::typeof(select_at_indices),s::AbstractArray,a::HyperReduction,args...)
  select_at_indices(s,a,args...)
end

function Arrays.return_cache(
  ::typeof(select_at_indices),
  s::Union{BlockArray,BlockParamArray},
  a::BlockHyperReduction,
  args...)

  @check size(blocks(s)) == size(a)
  @notimplementedif isempty(findall(a.touched))
  i = findfirst(a.touched)
  cache = return_cache(select_at_indices,blocks(s)[i],a[i],args...)
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function select_at_indices(s::Union{BlockArray,BlockParamArray},a::BlockHyperReduction,args...)
  s′ = return_cache(select_at_indices,s,a,args...)
  for i = eachindex(a)
    if a.touched[i]
      s′[i] = select_at_indices(blocks(s)[i],a[i],args...)
    end
  end
  return ArrayBlock(s′,a.touched)
end

function select_at_indices(s::ArrayContribution,a::AffineContribution)
  contribution(s.trians) do trian
    select_at_indices(s[trian],a[trian])
  end
end
