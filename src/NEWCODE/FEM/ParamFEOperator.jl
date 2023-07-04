abstract type ParamFEOperator{C<:OperatorType} <: GridapType end

# Default API

"""
Returns a `ParamOp` wrapper of the `ParamFEOperator`
"""
function get_algebraic_operator(feop::ParamFEOperator{C}) where C
  ParamOpFromFEOp{C}(feop)
end

# Specializations

"""
Parametric FE operator that is defined by a parametric weak form
"""
mutable struct ParamFEOperatorFromWeakForm{C<:OperatorType} <: ParamFEOperator{C}
  res::Function
  jac::Function
  assem::Assembler
  pspace::ParamSpace
  trial::Any
  test::FESpace
end

function ParamAffineFEOperator(res::Function,jac::Function,pspace,trial,test)
  # res(μ,u,v) = a(μ,u,v) - b(μ,v)
  # jac(μ,u,du,v) = a(μ,du,v)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOperatorFromWeakForm{Affine}(res,jac,assem,pspace,trial,test)
end

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOperatorFromWeakForm{Nonlinear}(res,jac,assem,pspace,trial,test)
end

function Gridap.FESpaces.SparseMatrixAssembler(
  trial::Union{ParamTrialFESpace,ParamMultiFieldTrialFESpace},
  test::FESpace)
  SparseMatrixAssembler(trial(nothing),test)
end

get_test(op::ParamFEOperatorFromWeakForm) = op.test
get_trial(op::ParamFEOperatorFromWeakForm) = op.trial
get_pspace(op::ParamFEOperatorFromWeakForm) = op.pspace
realization(op::ParamFEOperator,args...) = realization(op.pspace,args...)

function allocate_residual(
  op::ParamFEOperatorFromWeakForm,
  uh::CellField)

  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(realization(op),uh,v))
  allocate_vector(op.assem,vecdata)
end

function allocate_jacobian(
  op::ParamFEOperatorFromWeakForm,
  uh::CellField)

  Uμ = get_trial(op)
  U = Uμ(nothing)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(U,V,op.jac(realization(op),uh,du,v))
  allocate_matrix(op.assem,matdata)
end

function residual!(
  b::AbstractVector,
  op::ParamFEOperatorFromWeakForm,
  μ::AbstractVector,
  uh::CellField)

  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(μ,uh,v))
  assemble_vector!(b,op.assem,vecdata)
  b
end

function jacobian!(
  A::AbstractMatrix,
  op::ParamFEOperatorFromWeakForm,
  μ::AbstractVector,
  uh::CellField)

  Uμ = get_trial(op)
  U = Uμ(μ)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(U,V,op.jac(μ,uh,du,v))
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function _collect_trian_res(op::ParamFEOperator)
  μ = realization(op)
  uh = zero(op.test)
  v = get_fe_basis(op.test)
  veccontrib = op.res(μ,uh,v)
  collect_trian(veccontrib)
end

function _collect_trian_jac(op::ParamFEOperator)
  μ = realization(op)
  uh = zero(op.test)
  v = get_fe_basis(op.test)
  matcontrib = op.jac(μ,uh,v,v)
  collect_trian(matcontrib)
end

function get_single_field(
  op::ParamFEOperator{C},
  filter::Tuple{Vararg{Any}}) where C

  r_filter,c_filter = filter
  trial = op.trial
  test = op.test
  c_trial = trial[c_filter]
  r_test = test[r_filter]
  rc_assem = SparseMatrixAssembler(c_trial,r_test)
  ParamFEOperatorFromWeakForm{C}(
    op.res,
    op.jac,
    rc_assem,
    op.pspace,
    c_trial,
    r_test)
end
