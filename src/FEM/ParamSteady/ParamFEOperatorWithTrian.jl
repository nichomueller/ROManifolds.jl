"""
    struct ParamFEOperatorWithTrian{T} <: ParamFEOperator{T} end

Corresponds to a [`ParamFEOpFromWeakForm`](@ref) object, but in a triangulation
separation setting

"""
struct ParamFEOperatorWithTrian{T} <: ParamFEOperator{T}
  op::ParamFEOperator{T}
  trian_res::Tuple{Vararg{Triangulation}}
  trian_jac::Tuple{Vararg{Triangulation}}

  function ParamFEOperatorWithTrian(
    op::ParamFEOperator{T},
    trian_res::Tuple{Vararg{Triangulation}},
    trian_jac::Tuple{Vararg{Triangulation}}
    ) where T

    newop = set_triangulation(op,trian_res,trian_jac)
    new{T}(newop,trian_res,trian_jac)
  end
end

for f in (:ParamFEOperator,:LinearParamFEOperator)
  @eval begin
    function $f(res::Function,jac::Function,pspace::ParamSpace,
      trial::FESpace,test::FESpace,trian_res,trian_jac)

      op = $f(res,jac,pspace,trial,test)
      op_trian = ParamFEOperatorWithTrian(op,trian_res,trian_jac)
      return op_trian
    end
  end
end

FESpaces.get_test(op::ParamFEOperatorWithTrian) = get_test(op.op)
FESpaces.get_trial(op::ParamFEOperatorWithTrian) = get_trial(op.op)
get_param_space(op::ParamFEOperatorWithTrian) = get_param_space(op.op)
ODEs.get_res(op::ParamFEOperatorWithTrian) = get_res(op.op)
get_jac(op::ParamFEOperatorWithTrian) = get_jac(op.op)
ODEs.get_assembler(op::ParamFEOperatorWithTrian) = get_assembler(op.op)
IndexMaps.get_index_map(op::ParamFEOperatorWithTrian) = get_index_map(op.op)

function FESpaces.get_algebraic_operator(op::ParamFEOperatorWithTrian)
  ParamOpFromFEOpWithTrian(op)
end

# utils

function _set_triangulation_jac(
  jac::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newjac(μ,u,du,v,args...) = jac(μ,u,du,v,args...)
  newjac(μ,u,du,v) = newjac(μ,u,du,v,meas...)
  return newjac
end

function _set_triangulation_res(
  res::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newres(μ,u,v,args...) = res(μ,u,v,args...)
  newres(μ,u,v) = newres(μ,u,v,meas...)
  return newres
end

function set_triangulation(op::ParamFEOpFromWeakForm{T},trian_res,trian_jac) where T
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_res(op.res,trian_res,polyn_order)
  newjac = _set_triangulation_jac(op.jac,trian_jac,polyn_order)
  ParamFEOpFromWeakForm{T}(newres,newjac,op.pspace,op.assem,op.index_map,op.trial,op.test)
end

"""
    set_triangulation(op::ParamFEOperatorWithTrian,trian_res,trian_jac) -> ParamFEOperator

Two tuples of triangulations `trian_res` and `trian_jac` are substituted,
respectively, in the residual and jacobian of a ParamFEOperatorWithTrian, and
the resulting ParamFEOperator is returned

"""

function set_triangulation(
  op::ParamFEOperatorWithTrian,
  trian_res=op.trian_res,
  trian_jac=op.trian_jac)

  set_triangulation(op.op,trian_res,trian_jac)
end

"""
    change_triangulation(op::ParamFEOperatorWithTrian,trian_res,trian_jac) -> ParamFEOperatorWithTrian

Replaces the old triangulations relative to the residual and jacobian in `op` with
two new tuples `trian_res` and `trian_jac`, and returns the resulting ParamFEOperatorWithTrian

"""
function change_triangulation(op::ParamFEOperatorWithTrian,trian_res,trian_jac)
  newtrian_res = order_triangulations(op.trian_res,trian_res)
  newtrian_jac = order_triangulations(op.trian_jac,trian_jac)
  newop = set_triangulation(op,newtrian_res,newtrian_jac)
  ParamFEOperatorWithTrian(newop,newtrian_res,newtrian_jac)
end
