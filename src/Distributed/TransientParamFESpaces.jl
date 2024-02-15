Arrays.evaluate(U::DistributedSingleFieldFESpace,::Nothing,::Nothing) = U

(U::DistributedSingleFieldFESpace)(μ,t) = U

function Algebra.allocate_jacobian(
  op::TransientParamFEOperator,
  r::TransientParamRealization,
  duh::Union{GridapDistributed.DistributedCellField,GridapDistributed.DistributedMultiFieldFEFunction},
  cache)

  _matdata_jacobians = TransientFETools.fill_initial_jacobians(op,r,duh)
  matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
  assem = FEM.get_param_assembler(op.assem,r)
  allocate_matrix(assem,matdata)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::TransientDistributedCellField,
  γ::Tuple{Vararg{Real}},
  cache)

  _matdata_jacobians = TransientFETools.fill_jacobians(op,r,xh,γ)
  matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
  assem = FEM.get_param_assembler(op.assem,r)
  assemble_matrix_add!(A,assem,matdata)
  A
end

# function Algebra.allocate_residual(
#   op::TransientParamFEOperatorWithTrian,
#   r::TransientParamRealization,
#   duh::Union{GridapDistributed.DistributedCellField,GridapDistributed.DistributedMultiFieldFEFunction},
#   cache)

#   test = get_test(op)
#   v = get_fe_basis(test)
#   dxh = ()
#   for i in 1:get_order(op)
#     dxh = (dxh...,duh)
#   end
#   xh = TransientCellField(duh,dxh)
#   dc = op.op.res(get_params(r),get_times(r),xh,v)
#   assem = FEM.get_param_assembler(op.op.assem,r)
#   map(local_views(test),local_views(assem),local_views(dc),local_views.(op.trian_res)...
#   ) do test,assem,dc,(trians...)
#     b = array_contribution()
#     for trian in trians
#       vecdata = FEM.collect_cell_vector_for_trian(test,dc,trian)
#       b[trian] = allocate_vector(assem,vecdata)
#     end
#     b
#   end
# end

# function Algebra.allocate_jacobian(
#   op::TransientParamFEOperatorWithTrian,
#   r::TransientParamRealization,
#   duh::Union{GridapDistributed.DistributedCellField,GridapDistributed.DistributedMultiFieldFEFunction},
#   cache)

#   dxh = ()
#   for i in 1:get_order(op)
#     dxh = (dxh...,duh)
#   end
#   xh = TransientCellField(duh,dxh)
#   trial = evaluate(get_trial(op),nothing)
#   test = get_test(op)
#   u = get_trial_fe_basis(trial)
#   v = get_fe_basis(test)
#   assem = FEM.get_param_assembler(op.op.assem,r)

#   A = ()
#   for i = 1:get_order(op)+1
#     Ai = array_contribution()
#     dc = op.op.jacs[i](get_params(r),get_times(r),xh,u,v)
#     triani = op.trian_jacs[i]
#     Ai = map(local_views(trial),local_views(test),local_views(assem),local_views(dc),local_views.(triani)...
#     ) do trial,test,assem,dc,(trians...)
#       Ai = array_contribution()
#       for trian in trians
#         matdata = FEM.collect_cell_matrix_for_trian(trial,test,dc,trian)
#         Ai[trian] = allocate_matrix(assem,matdata)
#       end
#       Ai
#     end
#     A = (A...,Ai)
#   end
#   A
# end
