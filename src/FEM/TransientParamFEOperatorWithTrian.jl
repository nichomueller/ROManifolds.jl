# interface to accommodate the separation of terms depending on the triangulation

abstract type TransientParamFEOperatorWithTrian{T<:ODEParamOperatorType} <: TransientParamFEOperator{T} end

function FESpaces.get_algebraic_operator(feop::TransientParamFEOperatorWithTrian)
  ODEParamOpFromTFEOpWithTrian(feop)
end

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
  induced_norm::Function,
  tpspace::TransientParamSpace,
  assem::Assembler,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  trian_res,
  trian_jacs...)

  op = TransientParamFEOpFromWeakForm(res,jacs,induced_norm,tpspace,assem,
    trial,test,order)
  op_trian = TransientParamFEOpFromWeakFormWithTrian(op,trian_res,trian_jacs)
  return op_trian
end

function TransientParamSemilinearFEOpFromWeakForm(
  mass::Function,
  res::Function,
  jacs::Tuple{Vararg{Function}},
  constant_mass::Bool,
  induced_norm::Function,
  tpspace::TransientParamSpace,
  assem::Assembler,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  trian_res,
  trian_jacs...)

  op = TransientParamSemilinearFEOpFromWeakForm(mass,res,jacs,constant_mass,
    induced_norm,tpspace,assem,trial,test,order)
  op_trian = TransientParamFEOpFromWeakFormWithTrian(op,trian_res,trian_jacs)
  return op_trian
end

function TransientParamLinearFEOpFromWeakForm(
  forms::Tuple{Vararg{Function}},
  res::Function,
  jacs::Tuple{Vararg{Function}},
  constant_forms::Tuple{Vararg{Bool}},
  induced_norm::Function,
  tpspace::TransientParamSpace,
  assem::Assembler,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  trian_res,
  trian_jacs...)

  op = TransientParamLinearFEOpFromWeakForm(forms,res,jacs,constant_forms,
    induced_norm,tpspace,assem,trial,test,order)
  op_trian = TransientParamFEOpFromWeakFormWithTrian(op,trian_res,trian_jacs)
  return op_trian
end

FESpaces.get_test(op::TransientParamFEOpFromWeakFormWithTrian) = get_test(op.op)
FESpaces.get_trial(op::TransientParamFEOpFromWeakFormWithTrian) = get_trial(op.op)
Polynomials.get_order(op::TransientParamFEOpFromWeakFormWithTrian) = get_order(op.op)
ODEs.get_res(op::TransientParamFEOpFromWeakFormWithTrian) = get_res(op.op)
ODEs.get_jacs(op::TransientParamFEOpFromWeakFormWithTrian) = get_jacs(op.op)
ODEs.get_forms(op::TransientParamFEOpFromWeakFormWithTrian) = get_forms(op.op)
ODEs.get_assembler(op::TransientParamFEOpFromWeakFormWithTrian) = get_assembler(op.op)
realization(op::TransientParamFEOpFromWeakFormWithTrian;kwargs...) = realization(op.op;kwargs...)
get_induced_norm(op::TransientParamFEOpFromWeakFormWithTrian) = get_induced_norm(op.op)

function assemble_norm_matrix(op::TransientParamFEOpFromWeakFormWithTrian)
  assemble_norm_matrix(op.op)
end

function ODEs.get_assembler(op::TransientParamFEOpFromWeakFormWithTrian,r::TransientParamRealization)
  get_assembler(op.op,r)
end

struct TransientParamSaddlePointFEOpWithTrian{T,N} <: TransientParamFEOperatorWithTrian{T}
  op::TransientParamSaddlePointFEOp{T}
  trian_res::Tuple{Vararg{Triangulation}}
  trian_jacs::NTuple{N,Tuple{Vararg{Triangulation}}}

  function TransientParamSaddlePointFEOpWithTrian(
    op::TransientParamSaddlePointFEOp{T},
    trian_res::Tuple{Vararg{Triangulation}},
    trian_jacs::NTuple{N,Tuple{Vararg{Triangulation}}}) where {T,N}

    newop = set_triangulation(op,trian_res,trian_jacs)
    new{T,N}(newop,trian_res,trian_jacs)
  end
end

function TransientParamFEOpFromWeakForm(
  res::Function,
  jacs::Tuple{Vararg{Function}},
  induced_norm::Function,
  tpspace::TransientParamSpace,
  assem::Assembler,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  coupling::Function,
  trian_res,
  trian_jacs...)

  saddlep_op = TransientParamFEOpFromWeakForm(res,jacs,induced_norm,tpspace,assem,
    trial,test,order,coupling)
  saddlep_op_trian = TransientParamSaddlePointFEOpWithTrian(saddlep_op,trian_res,trian_jacs)
  return saddlep_op_trian
end

function TransientParamSemilinearFEOpFromWeakForm(
  mass::Function,
  res::Function,
  jacs::Tuple{Vararg{Function}},
  constant_mass::Bool,
  induced_norm::Function,
  tpspace::TransientParamSpace,
  assem::Assembler,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  coupling::Function,
  trian_res,
  trian_jacs...)

  saddlep_op = TransientParamSemilinearFEOpFromWeakForm(mass,res,jacs,constant_mass,
    induced_norm,tpspace,assem,trial,test,order,coupling)
  saddlep_op_trian = TransientParamSaddlePointFEOpWithTrian(saddlep_op,trian_res,trian_jacs)
  return saddlep_op_trian
end

function TransientParamLinearFEOpFromWeakForm(
  forms::Tuple{Vararg{Function}},
  res::Function,
  jacs::Tuple{Vararg{Function}},
  constant_forms::Tuple{Vararg{Bool}},
  induced_norm::Function,
  tpspace::TransientParamSpace,
  assem::Assembler,
  trial::FESpace,
  test::FESpace,
  order::Integer,
  coupling::Function,
  trian_res,
  trian_jacs...)

  saddlep_op = TransientParamLinearFEOpFromWeakForm(forms,res,jacs,constant_forms,
    induced_norm,tpspace,assem,trial,test,order,coupling)
  saddlep_op_trian = TransientParamSaddlePointFEOpWithTrian(saddlep_op,trian_res,trian_jacs)
  return saddlep_op_trian
end

FESpaces.get_test(op::TransientParamSaddlePointFEOpWithTrian) = get_test(op)
FESpaces.get_trial(op::TransientParamSaddlePointFEOpWithTrian) = get_trial(op)
Polynomials.get_order(op::TransientParamSaddlePointFEOpWithTrian) = get_order(op)
ODEs.get_res(op::TransientParamSaddlePointFEOpWithTrian) = get_res(op)
ODEs.get_jacs(op::TransientParamSaddlePointFEOpWithTrian) = get_jacs(op)
ODEs.get_forms(op::TransientParamSaddlePointFEOpWithTrian) = get_forms(op)
ODEs.get_assembler(op::TransientParamSaddlePointFEOpWithTrian) = get_assembler(op)
realization(op::TransientParamSaddlePointFEOpWithTrian;kwargs...) = realization(op;kwargs...)
get_induced_norm(op::TransientParamSaddlePointFEOpWithTrian) = get_induced_norm(op)
get_coupling(op::TransientParamSaddlePointFEOpWithTrian) = get_coupling(op)

function assemble_norm_matrix(op::TransientParamSaddlePointFEOpWithTrian)
  assemble_norm_matrix(op.op)
end

function assemble_coupling_matrix(op::TransientParamSaddlePointFEOpWithTrian)
  assemble_coupling_matrix(op.op)
end

function ODEs.get_assembler(op::TransientParamSaddlePointFEOpWithTrian,r::TransientParamRealization)
  get_assembler(op.op,r)
end

# utils

function _set_triangulation_jac(
  jac::Function,
  trian_jac::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian_jac,degree)
  newjac(μ,t,u,du,v,args...) = jac(μ,t,u,du,v,args...)
  newjac(μ,t,u,du,v) = newjac(μ,t,u,du,v,meas...)
  return newjac
end

function _set_triangulation_jacs(
  jacs::NTuple{N,Function},
  trian_jacs::NTuple{N,Tuple{Vararg{Triangulation}}},
  order) where N

  newjacs = ()
  for (jac,trians) in zip(jacs,trian_jacs)
    newjacs = (newjacs...,_set_triangulation_jac(jac,trians,order))
  end
  return newjacs
end

function _set_triangulation_form(
  res::Function,
  trian_res::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian_res,degree)
  newres(μ,t,u,v,args...) = res(μ,t,u,v,args...)
  newres(μ,t,u,v) = newres(μ,t,u,v,meas...)
  return newres
end

function _set_triangulation_forms(
  forms::NTuple{N,Function},
  trian_forms::NTuple{N,Tuple{Vararg{Triangulation}}},
  order) where N

  newforms = ()
  for (form,trians) in zip(forms,trian_forms)
    newforms = (newforms...,_set_triangulation_form(form,trians,order))
  end
  return newforms
end

function get_polynomial_order(basis,::DiscreteModel)
  cell_basis = get_data(basis)
  shapefuns = first(cell_basis.values).fields
  orders = get_order(shapefuns)
  first(orders)
end

function get_polynomial_order(basis,::CartesianDiscreteModel)
  cell_basis = get_data(basis)
  shapefun = first(cell_basis).fields
  get_order(shapefun)
end

get_polynomial_order(basis,trian::Triangulation) = get_polynomial_order(basis,get_background_model(trian))
get_polynomial_order(fs::SingleFieldFESpace) = get_polynomial_order(get_fe_basis(fs),get_triangulation(fs))
get_polynomial_order(fs::MultiFieldFESpace) = maximum(map(get_polynomial_order,fs.spaces))

function set_triangulation(op::TransientParamFEOpFromWeakForm,trian_res,trian_jacs)
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_form(op.res,trian_res,polyn_order)
  newjacs = _set_triangulation_jacs(op.jacs,trian_jacs,polyn_order)
  TransientParamFEOpFromWeakForm(
    newres,newjacs,op.induced_norm,op.tpspace,op.assem,op.trial,op.test,op.order)
end

function set_triangulation(op::TransientParamSemilinearFEOpFromWeakForm,trian_res,trian_jacs)
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_form(op.res,trian_res,polyn_order)
  newmass = _set_triangulation_form(op.mass,trian_jacs[end],polyn_order)
  newjacs = _set_triangulation_jacs(op.jacs,trian_jacs,polyn_order)
  TransientParamSemilinearFEOpFromWeakForm(
    newmass,newres,newjacs,op.constant_mass,op.induced_norm,op.tpspace,op.assem,
    op.trial,op.test,op.order)
end

function set_triangulation(op::TransientParamLinearFEOpFromWeakForm,trian_res,trian_jacs)
  polyn_order = get_polynomial_order(op.test)
  newres = _set_triangulation_form(op.res,trian_res,polyn_order)
  newforms = _set_triangulation_forms(op.forms,trian_jacs,polyn_order)
  newjacs = _set_triangulation_jacs(op.jacs,trian_jacs,polyn_order)
  TransientParamLinearFEOpFromWeakForm(
    newforms,newres,newjacs,op.constant_forms,op.induced_norm,op.tpspace,
    op.assem,op.trial,op.test,op.order)
end

function set_triangulation(op::TransientParamSaddlePointFEOp,trian_res,trian_jacs)
  newop = set_triangulation(op.op,trian_res,trian_jacs)
  TransientParamSaddlePointFEOp(newop,op.coupling)
end

function set_triangulation(
  op::TransientParamFEOperatorWithTrian,
  trian_res=op.trian_res,
  trian_jacs=op.trian_jacs)

  set_triangulation(op.op,trian_res,trian_jacs)
end

function change_triangulation(op::TransientParamFEOperatorWithTrian,trian_res,trian_jacs)
  newtrian_res = order_triangulations(op.trian_res,trian_res)
  newtrian_jacs = order_triangulations.(op.trian_jacs,trian_jacs)
  newop = set_triangulation(op,newtrian_res,newtrian_jacs)
  TransientParamFEOperatorWithTrian(newop,newtrian_res,newtrian_jacs)
end
