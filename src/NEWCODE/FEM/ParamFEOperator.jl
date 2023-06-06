abstract type ParamFEOperator{C<:OperatorType} <: GridapType end

# Default API

"""
Returns a `ParamOp` wrapper of the `ParamFEOperator`
"""
function Gridap.ODEs.TransientFETools.get_algebraic_operator(feop::ParamFEOperator{C}) where C
  ParamOpFromFEOp{C}(feop)
end

# Specializations

"""
Parametric FE operator that is defined by a parametric weak form
"""
struct ParamFEOperatorFromWeakForm{C<:OperatorType} <: ParamFEOperator{C}
  res::Function
  jac::Function
  assem::Assembler
  pspace::ParamSpace
  trial::Any
  test::FESpace
end

function ParamAffineFEOperator(a::Function,b::Function,pspace,trial,test)
  res(μ,u,v) = a(μ,u,v) - b(μ,v)
  jac(μ,u,du,v) = a(μ,du,v)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOperatorFromWeakForm{Affine}(res,jac,assem,pspace,trial,test)
end

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOperatorFromWeakForm{Nonlinear}(res,jac,assem,pspace,trial,test)
end

function Gridap.ODEs.TransientFETools.SparseMatrixAssembler(
  trial::Union{ParamTrialFESpace,ParamMultiFieldTrialFESpace},
  test::FESpace)
  SparseMatrixAssembler(Gridap.evaluate(trial,nothing),test)
end

Gridap.ODEs.TransientFETools.get_assembler(op::ParamFEOperatorFromWeakForm) = op.assem
Gridap.ODEs.TransientFETools.get_test(op::ParamFEOperatorFromWeakForm) = op.test
Gridap.FESpaces.get_trial(op::ParamFEOperatorFromWeakForm) = op.trial

function Gridap.ODEs.TransientFETools.allocate_residual(
  op::ParamFEOperatorFromWeakForm,
  uh::CellField)

  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(realization(op),uh,v))
  allocate_vector(op.assem,vecdata)
end

function Gridap.ODEs.TransientFETools.allocate_jacobian(
  op::ParamFEOperatorFromWeakForm,
  uh::CellField)

  Uμ = get_trial(op)
  U = Gridap.evaluate(Uμ,nothing)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(U,V,op.jac(realization(op),uh,du,v))
  allocate_matrix(op.assem,matdata)
end

function Gridap.ODEs.TransientFETools.residual!(
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

function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::ParamFEOperatorFromWeakForm,
  μ::AbstractVector,
  uh::CellField)

  Uμ = get_trial(op)
  U = Gridap.evaluate(Uμ,μ)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(U,V,op.jac(μ,uh,du,v))
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

get_pspace(op::ParamFEOperatorFromWeakForm) = op.pspace
realization(op::ParamFEOperator,args...) = realization(op.pspace,args...)
get_Ns(space::FESpace) = space.nfree
get_Ns(space::ZeroMeanFESpace) = space.space.space.nfree-1
get_Ns(space::MultiFieldFESpace) = get_Ns.(space)
get_Ns(op::ParamFEOperator) = get_Ns(op.test)
