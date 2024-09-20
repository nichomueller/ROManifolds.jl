Arrays.evaluate(U::DistributedSingleFieldFESpace,::Nothing,::Nothing) = U

(U::DistributedSingleFieldFESpace)(μ,t) = U

function Algebra.allocate_jacobian(
  op::TransientParamFEOperator,
  r::TransientRealization,
  duh::Union{GridapDistributed.DistributedCellField,GridapDistributed.DistributedMultiFieldFEFunction},
  cache)

  _matdata_jacobians = TransientFETools.fill_initial_jacobians(op,r,duh)
  matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
  assem = FEM.get_param_assembler(op.assem,r)
  allocate_matrix(assem,matdata)
end

function ODEs.jacobians!(
  A::AbstractMatrix,
  op::TransientParamFEOpFromWeakForm,
  r::TransientRealization,
  xh::TransientDistributedCellField,
  γ::Tuple{Vararg{Real}},
  cache)

  _matdata_jacobians = TransientFETools.fill_jacobians(op,r,xh,γ)
  matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
  assem = FEM.get_param_assembler(op.assem,r)
  assemble_matrix_add!(A,assem,matdata)
  A
end

function Algebra.allocate_residual(
  op::TransientParamFEOperatorWithTrian,
  r::TransientRealization,
  duh::Union{GridapDistributed.DistributedCellField,GridapDistributed.DistributedMultiFieldFEFunction},
  cache)

  FEM._allocate_residual(b,op,r,duh,cache)
end

function Algebra.allocate_jacobian(
  op::TransientParamFEOperatorWithTrian,
  r::TransientRealization,
  duh::Union{GridapDistributed.DistributedCellField,GridapDistributed.DistributedMultiFieldFEFunction},
  cache)

  FEM._allocate_jacobian(A,op,r,duh,cache)
end
