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
  assem = get_param_assembler(op.op,r)

  trian_res = get_domains_res(op.op)
  res = get_res(op.op)
  dc = res(r,uh,v)

  map(trian_res) do strian
    b_trian = b.fecache[strian]
    i_trian = get_integration_domain(op.rhs[strian])
    scell_vec = get_contribution(dc,strian)
    cell_vec,trian = move_contributions(scell_vec,strian)
    @assert ndims(eltype(cell_vec)) == 1
    cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
    vecdata = [cell_vec_r],[i_trian.cell_irows]
    assemble_hr_vector_add!(b_trian,assem,vecdata)
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
  assem = get_param_assembler(op.op,r)

  trian_jac = get_domains_jac(op.op)
  jac = get_jac(op.op)
  dc = jac(r,uh,du,v)

  map(trian_jac) do strian
    A_trian = A.fecache[strian]
    i_trian = get_integration_domain(op.lhs[strian])
    scell_mat = get_contribution(dc,strian)
    cell_mat,trian = move_contributions(scell_mat,strian)
    @assert ndims(eltype(cell_mat)) == 2
    cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
    cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
    matdata = [cell_mat_rc],[i_trian.cell_irows],[i_trian.cell_icols]
    assemble_hr_matrix_add!(A_trian,assem,matdata)
  end

  inv_project!(A,op.lhs)
end

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
