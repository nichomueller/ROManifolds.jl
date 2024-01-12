struct Nonaffine <: OperatorType end

abstract type PODEOperator{C<:OperatorType} <: ODEOperator{C} end

const AffineODEOperator = ODEOperator{Affine}
const NonaffineODEOperator = ODEOperator{Nonaffine}

struct PODEOpFromFEOp{C} <: PODEOperator{C}
  feop::TransientPFEOperator{C}
end
