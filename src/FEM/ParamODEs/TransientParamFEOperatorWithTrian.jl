"""
    abstract type TransientParamFEOperatorWithTrian{T<:ODEParamOperatorType} <:
      TransientParamFEOperator{T} end

Interface to accommodate the separation of terms in the problem's weak formulation
depending on the triangulation on which the integration occurs. When employing
a TransientParamFEOperatorWithTrian, the residual and jacobian are returned as
[`Contribution`](@ref) objects, instead of standard arrays. To correctly define
an instance of TransientParamFEOperatorWithTrian, one needs to:
- provide the integration domains of the residual and jacobian, i.e. their
  respective triangulations
- define the residual and jacobian as functions of the Measure objects corresponding
  to the aforementioned triangulations

Subtypes:

- [`TransientParamFEOpFromWeakFormWithTrian`](@ref)
- [`LinearNonlinearTransientParamFEOperatorWithTrian`](@ref)

"""
abstract type TransientParamFEOperatorWithTrian{T<:ODEParamOperatorType} <: TransientParamFEOperator{T} end

function FESpaces.get_algebraic_operator(feop::TransientParamFEOperatorWithTrian)
  ODEParamOpFromTFEOpWithTrian(feop)
end

"""
    struct TransientParamFEOpFromWeakFormWithTrian{T,N} <: TransientParamFEOperatorWithTrian{T} end

Corresponds to a [`TransientParamFEOpFromWeakForm`](@ref) object, but in a triangulation
separation setting. `N` corresponds to the length of the jacobians in the transient
problem, i.e. the order of the time derivative

"""
struct TransientParamFEOpFromWeakFormWithTrian{T,N} <: TransientParamFEOperatorWithTrian{T}
  op::TransientParamFEOperator{T}
  trian_res::Tuple{Vararg{Triangulation}}
  trian_jacs::NTuple{N,Tuple{Vararg{Triangulation}}}

  function TransientParamFEOpFromWeakFormWithTrian(
    op::TransientParamFEOperator{T},
    trian_res::Tuple{Vararg{Triangulation}},
    trian_jacs::NTuple{N,Tuple{Vararg{Triangulation}}}) where {T,N}

    newop = set_triangulation(op,trian_res,trian_jacs)
    new{T,N}(newop,trian_res,trian_jacs)
  end
end

function TransientParamFEOpFromWeakForm(
  res::Function,
  jacs::Tuple{Vararg{Function}},
  tpspace::TransientParamSpace,
  assem::Assembler,
  index_map::FEOperatorIndexMap,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  trian_res,
  trian_jacs...)

  op = TransientParamFEOpFromWeakForm(res,jacs,tpspace,assem,index_map,trial,test,order)
  op_trian = TransientParamFEOpFromWeakFormWithTrian(op,trian_res,trian_jacs)
  return op_trian
end

function TransientParamSemilinearFEOpFromWeakForm(
  mass::Function,
  res::Function,
  jacs::Tuple{Vararg{Function}},
  constant_mass::Bool,
  tpspace::TransientParamSpace,
  assem::Assembler,
  index_map::FEOperatorIndexMap,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  trian_res,
  trian_jacs...)

  op = TransientParamSemilinearFEOpFromWeakForm(mass,res,jacs,constant_mass,tpspace,assem,index_map,trial,test,order)
  op_trian = TransientParamFEOpFromWeakFormWithTrian(op,trian_res,trian_jacs)
  return op_trian
end

function TransientParamLinearFEOpFromWeakForm(
  forms::Tuple{Vararg{Function}},
  res::Function,
  jacs::Tuple{Vararg{Function}},
  constant_forms::Tuple{Vararg{Bool}},
  tpspace::TransientParamSpace,
  assem::Assembler,
  index_map::FEOperatorIndexMap,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  trian_res,
  trian_jacs...)

  op = TransientParamLinearFEOpFromWeakForm(forms,res,jacs,constant_forms,tpspace,assem,index_map,trial,test,order)
  op_trian = TransientParamFEOpFromWeakFormWithTrian(op,trian_res,trian_jacs)
  return op_trian
end

FESpaces.get_test(op::TransientParamFEOpFromWeakFormWithTrian) = get_test(op.op)
FESpaces.get_trial(op::TransientParamFEOpFromWeakFormWithTrian) = get_trial(op.op)
ReferenceFEs.get_order(op::TransientParamFEOpFromWeakFormWithTrian) = get_order(op.op)
ODEs.get_res(op::TransientParamFEOpFromWeakFormWithTrian) = get_res(op.op)
ODEs.get_jacs(op::TransientParamFEOpFromWeakFormWithTrian) = get_jacs(op.op)
ODEs.get_assembler(op::TransientParamFEOpFromWeakFormWithTrian) = get_assembler(op.op)
IndexMaps.get_index_map(op::TransientParamFEOpFromWeakFormWithTrian) = get_index_map(op.op)
ParamDataStructures.realization(op::TransientParamFEOpFromWeakFormWithTrian;kwargs...) = realization(op.op;kwargs...)

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
  trians::NTuple{N,Tuple{Vararg{Triangulation}}},
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

function _set_triangulation_forms(
  forms::NTuple{N,Function},
  trians::NTuple{N,Tuple{Vararg{Triangulation}}},
  order) where N

  newforms = ()
  for (form,trian) in zip(forms,trians)
    newforms = (newforms...,_set_triangulation_form(form,trian,order))
  end
  return newforms
end

function ParamSteady.set_triangulation(op::TransientParamFEOpFromWeakForm,trian_res,trian_jacs)
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_form(op.res,trian_res,polyn_order)
  newjacs = _set_triangulation_jacs(op.jacs,trian_jacs,polyn_order)
  TransientParamFEOpFromWeakForm(
    newres,newjacs,op.tpspace,op.assem,op.index_map,op.trial,op.test,op.order)
end

function ParamSteady.set_triangulation(op::TransientParamSemilinearFEOpFromWeakForm,trian_res,trian_jacs)
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_form(op.res,trian_res,polyn_order)
  newmass = _set_triangulation_form(op.mass,trian_jacs[end],polyn_order)
  newjacs = _set_triangulation_jacs(op.jacs,trian_jacs,polyn_order)
  TransientParamSemilinearFEOpFromWeakForm(
    newmass,newres,newjacs,op.constant_mass,op.tpspace,op.assem,
    op.index_map,op.trial,op.test,op.order)
end

function ParamSteady.set_triangulation(op::TransientParamLinearFEOpFromWeakForm,trian_res,trian_jacs)
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_form(op.res,trian_res,polyn_order)
  newforms = _set_triangulation_forms(op.forms,trian_jacs,polyn_order)
  newjacs = _set_triangulation_jacs(op.jacs,trian_jacs,polyn_order)
  TransientParamLinearFEOpFromWeakForm(
    newforms,newres,newjacs,op.constant_forms,op.tpspace,
    op.assem,op.index_map,op.trial,op.test,op.order)
end

function ParamSteady.set_triangulation(
  op::TransientParamFEOperatorWithTrian,
  trian_res=op.trian_res,
  trian_jacs=op.trian_jacs)

  set_triangulation(op.op,trian_res,trian_jacs)
end

function ParamSteady.change_triangulation(op::TransientParamFEOpFromWeakFormWithTrian,trian_res,trian_jacs)
  newtrian_res = order_triangulations(op.trian_res,trian_res)
  newtrian_jacs = order_triangulations.(op.trian_jacs,trian_jacs)
  newop = set_triangulation(op,newtrian_res,newtrian_jacs)
  TransientParamFEOpFromWeakFormWithTrian(newop,newtrian_res,newtrian_jacs)
end
