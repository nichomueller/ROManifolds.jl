"""
    struct TransientParamFEOperatorWithTrian{T,N} <: TransientParamFEOperator{T} end

Corresponds to a [`TransientParamFEOpFromWeakForm`](@ref) object, but in a triangulation
separation setting. `N` corresponds to the length of the jacobians in the transient
problem, i.e. the order of the time derivative

"""
struct TransientParamFEOperatorWithTrian{T,N} <: TransientParamFEOperator{T}
  op::TransientParamFEOperator{T}
  trian_res::Tuple{Vararg{Triangulation}}
  trian_jacs::NTuple{N,Tuple{Vararg{Triangulation}}}

  function TransientParamFEOperatorWithTrian(
    op::TransientParamFEOperator{T},
    trian_res::Tuple{Vararg{Triangulation}},
    trian_jacs::NTuple{N,Tuple{Vararg{Triangulation}}}) where {T,N}

    newop = set_triangulation(op,trian_res,trian_jacs)
    new{T,N}(newop,trian_res,trian_jacs)
  end
end

function TransientParamFEOpFromWeakForm(res::Function,jacs::Tuple{Vararg{Function}},
  tpspace::TransientParamSpace,assem::Assembler,index_map::FEOperatorIndexMap,
  trial::FESpace,test::FESpace,order::Integer,trian_res,trian_jacs...)

  op = TransientParamFEOpFromWeakForm(res,jacs,tpspace,assem,index_map,trial,test,order)
  op_trian = TransientParamFEOperatorWithTrian(op,trian_res,trian_jacs)
  return op_trian
end

function TransientParamLinearFEOpFromWeakForm(res::Function,jacs::Tuple{Vararg{Function}},
  constant_forms::Tuple{Vararg{Bool}},tpspace::TransientParamSpace,assem::Assembler,
  index_map::FEOperatorIndexMap,trial::FESpace,test::FESpace,order::Integer,trian_res,trian_jacs...)

  op = TransientParamLinearFEOpFromWeakForm(res,jacs,constant_forms,tpspace,assem,index_map,trial,test,order)
  op_trian = TransientParamFEOperatorWithTrian(op,trian_res,trian_jacs)
  return op_trian
end

function FESpaces.get_algebraic_operator(feop::TransientParamFEOperatorWithTrian)
  ODEParamOpFromTFEOpWithTrian(feop)
end

FESpaces.get_test(op::TransientParamFEOperatorWithTrian) = get_test(op.op)
FESpaces.get_trial(op::TransientParamFEOperatorWithTrian) = get_trial(op.op)
ParamSteady.get_param_space(op::TransientParamFEOperatorWithTrian) = get_param_space(op.op)
Polynomials.get_order(op::TransientParamFEOperatorWithTrian) = get_order(op.op)
ODEs.get_res(op::TransientParamFEOperatorWithTrian) = get_res(op.op)
ODEs.get_jacs(op::TransientParamFEOperatorWithTrian) = get_jacs(op.op)
ODEs.get_assembler(op::TransientParamFEOperatorWithTrian) = get_assembler(op.op)
ODEs.is_form_constant(op::TransientParamFEOperatorWithTrian,k::Integer) = is_form_constant(op.op,k)
IndexMaps.get_index_map(op::TransientParamFEOperatorWithTrian) = get_index_map(op.op)

# utils

function _set_triangulation_jac(
  jac::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newjac(μ,t,u,du,v,args...) = jac(μ,t,u,du,v,args...)
  newjac(μ,t,u,du,v) = newjac(μ,t,u,du,v,meas...)
  return newjac
end

function _set_triangulation_jacs(
  jacs::NTuple{N,Function},
  trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}},
  order) where N

  newjacs = ()
  for (jac,trian) in zip(jacs,trians)
    newjacs = (newjacs...,_set_triangulation_jac(jac,trian,order))
  end
  return newjacs
end

function _set_triangulation_form(
  res::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newres(μ,t,u,v,args...) = res(μ,t,u,v,args...)
  newres(μ,t,u,v) = newres(μ,t,u,v,meas...)
  return newres
end

function ParamSteady.set_triangulation(op::TransientParamFEOpFromWeakForm,trian_res,trian_jacs)
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_form(op.res,trian_res,polyn_order)
  newjacs = _set_triangulation_jacs(op.jacs,trian_jacs,polyn_order)
  TransientParamFEOpFromWeakForm(
    newres,newjacs,op.tpspace,op.assem,op.index_map,op.trial,op.test,op.order)
end

function ParamSteady.set_triangulation(op::TransientParamLinearFEOpFromWeakForm,trian_res,trian_jacs)
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_form(op.res,trian_res,polyn_order)
  newjacs = _set_triangulation_jacs(op.jacs,trian_jacs,polyn_order)
  TransientParamLinearFEOpFromWeakForm(
    newres,newjacs,op.constant_forms,op.tpspace,
    op.assem,op.index_map,op.trial,op.test,op.order)
end

function ParamSteady.set_triangulation(
  op::TransientParamFEOperatorWithTrian,
  trian_res=op.trian_res,
  trian_jacs=op.trian_jacs)

  set_triangulation(op.op,trian_res,trian_jacs)
end

function ParamSteady.change_triangulation(op::TransientParamFEOperatorWithTrian,trian_res,trian_jacs)
  newtrian_res = order_triangulations(op.trian_res,trian_res)
  newtrian_jacs = order_triangulations.(op.trian_jacs,trian_jacs)
  newop = set_triangulation(op,newtrian_res,newtrian_jacs)
  TransientParamFEOperatorWithTrian(newop,newtrian_res,newtrian_jacs)
end
