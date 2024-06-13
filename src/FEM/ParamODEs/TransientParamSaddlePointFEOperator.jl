# interface to deal with the inf-sup stability condition of saddle point problems

function TransientParamFEOpFromWeakForm(
  res::Function,
  jacs::Tuple{Vararg{Function}},
  induced_norm::Function,
  tpspace::TransientParamSpace,
  index_map::FEOperatorIndexMap,
  assem::Assembler,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  coupling::Function)

  op = TransientParamFEOpFromWeakForm(res,jacs,induced_norm,tpspace,assem,index_map,
    trial,test,order)
  saddlep_op = TransientParamSaddlePointFEOp(op,coupling)
  return saddlep_op
end

function TransientParamSemilinearFEOpFromWeakForm(
  mass::Function,
  res::Function,
  jacs::Tuple{Vararg{Function}},
  constant_mass::Bool,
  induced_norm::Function,
  tpspace::TransientParamSpace,
  assem::Assembler,
  index_map::FEOperatorIndexMap,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  coupling::Function)

  op = TransientParamSemilinearFEOpFromWeakForm(mass,res,jacs,constant_mass,
    induced_norm,tpspace,assem,index_map,trial,test,order)
  saddlep_op = TransientParamSaddlePointFEOp(op,coupling)
  return saddlep_op
end

function TransientParamLinearFEOpFromWeakForm(
  forms::Tuple{Vararg{Function}},
  res::Function,
  jacs::Tuple{Vararg{Function}},
  constant_forms::Tuple{Vararg{Bool}},
  induced_norm::Function,
  tpspace::TransientParamSpace,
  assem::Assembler,
  index_map::FEOperatorIndexMap,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  coupling::Function)

  op = TransientParamLinearFEOpFromWeakForm(forms,res,jacs,constant_forms,
    induced_norm,tpspace,assem,index_map,trial,test,order)
  saddlep_op = TransientParamSaddlePointFEOp(op,coupling)
  return saddlep_op
end

struct TransientParamSaddlePointFEOp{T<:ParamOperatorType} <: TransientParamFEOperator{T}
  op::TransientParamFEOperator{T}
  coupling::Function
end

FESpaces.get_test(op::TransientParamSaddlePointFEOp) = get_test(op.op)
FESpaces.get_trial(op::TransientParamSaddlePointFEOp) = get_trial(op.op)
ReferenceFEs.get_order(op::TransientParamSaddlePointFEOp) = get_order(op.op)
ODEs.get_res(op::TransientParamSaddlePointFEOp) = get_res(op.op)
ODEs.get_jacs(op::TransientParamSaddlePointFEOp) = get_jacs(op.op)
ODEs.get_assembler(op::TransientParamSaddlePointFEOp) = get_assembler(op.op)
IndexMaps.get_index_map(op::TransientParamSaddlePointFEOp) = get_index_map(op.op)
ParamDataStructures.realization(op::TransientParamSaddlePointFEOp;kwargs...) = realization(op.op;kwargs...)
get_induced_norm(op::TransientParamSaddlePointFEOp) = get_induced_norm(op.op)
get_coupling(op::TransientParamSaddlePointFEOp) = op.coupling
ODEs.is_form_constant(op::TransientParamSaddlePointFEOp,k::Integer) = is_form_constant(op.op,k)
