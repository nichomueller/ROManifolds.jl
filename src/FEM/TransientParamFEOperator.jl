"""
A parametric version of the `Gridap` `TransientFEOperator`
"""
abstract type TransientParamFEOperator{T<:OperatorType} <: GridapType end

function FESpaces.get_algebraic_operator(feop::TransientParamFEOperator{C}) where C
  ODEParamOpFromFEOp{C}(feop)
end

function TransientFETools.allocate_cache(op::TransientParamFEOperator)
  nothing
end

function TransientFETools.update_cache!(cache::Nothing,op::TransientParamFEOperator,r)
  nothing
end

function FESpaces.SparseMatrixAssembler(
  trial::Union{TransientTrialParamFESpace,TransientMultiFieldTrialParamFESpace},
  test::FESpace)
  SparseMatrixAssembler(trial(nothing),test)
end

function TransientFETools.rhs_error(μ,t,u,v)
  error("The \"rhs\" function is not defined for this TransientFEOperator.
  Please, try to use another type of TransientFEOperator that supports this
  functionality.")
end

"""
Transient FE operator that is defined by a transient Weak form
"""
struct TransientParamFEOperatorFromWeakForm{T<:OperatorType} <: TransientParamFEOperator{T}
  res::Function
  rhs::Function
  jacs::Tuple{Vararg{Function}}
  induced_norm::Function
  assem::Assembler
  tpspace::TransientParamSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function AffineTransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,tpspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOperatorFromWeakForm{Affine}(
    res,TransientFETools.rhs_error,(jac,),induced_norm,assem,tpspace,(trial,),test,0)
end

function AffineTransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,tpspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOperatorFromWeakForm{Affine}(
    res,TransientFETools.rhs_error,(jac,jac_t),induced_norm,assem,tpspace,(trial,∂t(trial)),test,1)
end

function TransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,tpspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOperatorFromWeakForm{Nonlinear}(
    res,TransientFETools.rhs_error,(jac,),induced_norm,assem,tpspace,(trial,),test,0)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,tpspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOperatorFromWeakForm{Nonlinear}(
    res,TransientFETools.rhs_error,(jac,jac_t),induced_norm,assem,tpspace,(trial,∂t(trial)),test,1)
end

FESpaces.get_test(op::TransientParamFEOperatorFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamFEOperatorFromWeakForm) = op.trials[1]
ReferenceFEs.get_order(op::TransientParamFEOperatorFromWeakForm) = op.order
realization(op::TransientParamFEOperatorFromWeakForm;kwargs...) = realization(op.tpspace;kwargs...)

function assemble_norm_matrix(op::TransientParamFEOperatorFromWeakForm)
  test = get_test(op)
  trial = evalute(get_trial(op),(nothing))
  assemble_matrix(op.induced_norm,trial,test)
end

function Algebra.allocate_residual(
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  uh::T,
  cache) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  dc = op.res(get_params(r),get_times(r),xh,v)
  assem = get_param_assembler(op.assem,r)
  vecdata = collect_cell_vector(test,dc)
  allocate_vector(assem,vecdata)
end

function Algebra.allocate_jacobian(
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  uh::CellField,
  cache)

  _matdata_jacobians = TransientFETools.fill_initial_jacobians(op,r,uh)
  matdata = TransientFETools._vcat_matdata(_matdata_jacobians)
  assem = get_param_assembler(op.assem,r)
  allocate_matrix(assem,matdata)
end

function Algebra.residual!(
  b::AbstractVector,
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  cache) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dc = op.res(get_params(r),get_times(r),xh,v)
  vecdata = collect_cell_vector(test,dc)
  assem = get_param_assembler(op.assem,r)
  assemble_vector!(b,assem,vecdata)
  b
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache) where T

  matdata = _matdata_jacobian(op,r,xh,i,γᵢ)
  assem = get_param_assembler(op.assem,r)
  assemble_matrix_add!(A,assem,matdata)
  A
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  γ::Tuple{Vararg{Real}},
  cache) where T

  _matdata_jacobians = TransientFETools.fill_jacobians(op,r,xh,γ)
  matdata = TransientFETools._vcat_matdata(_matdata_jacobians)
  assem = get_param_assembler(op.assem,r)
  assemble_matrix_add!(A,assem,matdata)
  A
end

function TransientFETools.fill_initial_jacobians(
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  uh::T) where T

  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  _matdata = ()
  for i in 1:get_order(op)+1
    _matdata = (_matdata...,_matdata_jacobian(op,r,xh,i,0.0))
  end
  return _matdata
end

function TransientFETools.fill_jacobians(
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  γ::Tuple{Vararg{Real}}) where T

  _matdata = ()
  for i in 1:get_order(op)+1
    if (γ[i] > 0.0)
      _matdata = (_matdata...,_matdata_jacobian(op,r,xh,i,γ[i]))
    end
  end
  return _matdata
end

function TransientFETools._matdata_jacobian(
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  i::Integer,
  γᵢ::Real) where T

  trial = evaluate(get_trial(op),nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = γᵢ*op.jacs[i](get_params(r),get_times(r),xh,u,v)
  collect_cell_matrix(trial,test,dc)
end

function TransientFETools.test_transient_fe_operator(op::TransientParamFEOperator,uh,μt)
  odeop = get_algebraic_operator(op)
  @test isa(odeop,ODEParamOperator)
  cache = allocate_cache(op)
  V = get_test(op)
  @test isa(V,FESpace)
  U = get_trial(op)
  U0 = U(μt)
  @test isa(U0,TrialParamFESpace)
  r = allocate_residual(op,μt,uh,cache)
  @test isa(r,ParamVector)
  xh = TransientCellField(uh,(uh,))
  residual!(r,op,μt,xh,cache)
  @test isa(r,ParamVector)
  J = allocate_jacobian(op,μt,uh,cache)
  @test isa(J,ParamMatrix)
  jacobian!(J,op,μt,xh,1,1.0,cache)
  @test isa(J,ParamMatrix)
  jacobian!(J,op,μt,xh,2,1.0,cache)
  @test isa(J,ParamMatrix)
  jacobians!(J,op,μt,xh,(1.0,1.0),cache)
  @test isa(J,ParamMatrix)
  cache = update_cache!(cache,op,μt)
  true
end
