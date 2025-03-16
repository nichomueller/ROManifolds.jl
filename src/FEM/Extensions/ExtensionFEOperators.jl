struct PartialFEOperator <: FEOperator

end

struct ExtensionFEOperator{A<:FEOperator,B<:FEOperator} <: FEOperator
  int_op::A
  ext_op::B
end

function ExtensionFEOperator(
  res::Function,
  jac::Function,
  trial::ExtensionFESpace,
  test::ExtensionFESpace
  )

  int_op = FEOperator(res,jac,get_internal_space(trial),get_internal_space(test))
  ext_op = AffineFEOperator()
end
