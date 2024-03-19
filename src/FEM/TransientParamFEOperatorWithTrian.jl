# interface to accommodate the separation of terms depending on the triangulation

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},induced_norm::Function,tpspace,trial,test,
  trian_res,trian_jacs,args...)

  op = TransientParamFEOperator(res,jacs,induced_norm,tpspace,trial,test,args...)
  TransientFEOperatorWithTrian(op,trian_res,trian_jacs)
end

function TransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,tpspace,trial,test,trian_res,trian_jacs,args...)

  op = TransientParamFEOperator(res,jac,induced_norm,tpspace,trial,test,args...)
  TransientFEOperatorWithTrian(op,trian_res,trian_jacs)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,tpspace,trial,test,
  trian_res,trian_jacs,args...)

  op = TransientParamFEOperator(res,jac,jac_t,induced_norm,tpspace,trial,test,args...)
  TransientFEOperatorWithTrian(op,trian_res,trian_jacs)
end

function TransientParamFEOperator(
  res::Function,induced_norm::Function,tpspace,trial,test,trian_res,trian_jacs,args...;kwargs...)

  @notimplemented "When building a TransientFEOperatorWithTrian, the jacobians
  must explicitly be defined"
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,jacs::Tuple{Vararg{Function}},
  induced_norm::Function,tpspace,trial,test,trian_res,trian_jacs,args...;kwargs...)

  op = TransientParamLinearFEOperator(forms,res,jacs,induced_norm,tpspace,trial,test,args...;kwargs...)
  TransientFEOperatorWithTrian(op,trian_res,trian_jacs)
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,jac::Function,induced_norm::Function,
  tpspace,trial,test,trian_res,trian_jacs,args...;kwargs...)

  op = TransientParamLinearFEOperator(forms,res,jac,induced_norm,tpspace,trial,test,args...;kwargs...)
  TransientFEOperatorWithTrian(op,trian_res,trian_jacs)
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,jac::Function,jac_t::Function,
  induced_norm::Function,tpspace,trial,test,trian_res,trian_jacs,args...;kwargs...)

  op = TransientParamLinearFEOperator(forms,res,jac,jac_t,induced_norm,tpspace,trial,test,args...;kwargs...)
  TransientFEOperatorWithTrian(op,trian_res,trian_jacs)
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,induced_norm::Function,
  tpspace,trial,test,trian_res,trian_jacs,args...;kwargs...)

  @notimplemented "When building a TransientFEOperatorWithTrian, the jacobians
  must explicitly be defined"
end

abstract type TransientParamFEOperatorWithTrian{T<:ODEParamOperatorType} <: TransientParamFEOperator{T} end

function FESpaces.get_algebraic_operator(feop::TransientParamFEOperatorWithTrian)
  ODEParamOpFromTFEOpWithTrian(feop)
end

struct TransientParamFEOpFromWeakFormWithTrian{T} <: TransientParamFEOperatorWithTrian{T}
  op::TransientParamFEOpFromWeakForm{T}
  trian_res::Tuple{Vararg{Triangulation}}
  trian_jacs::NTuple{2,Tuple{Vararg{Triangulation}}}
end

function TransientFEOperatorWithTrian(op::TransientParamFEOpFromWeakForm,trian_res,trian_jacs...)
  newop = set_triangulation(op,trian_res,trian_jacs)
  TransientParamFEOperatorWithTrian(newop,trian_res,trian_jacs)
end

FESpaces.get_test(op::TransientParamFEOpFromWeakFormWithTrian) = get_test(op)
FESpaces.get_trial(op::TransientParamFEOpFromWeakFormWithTrian) = get_trial(op)
Polynomials.get_order(op::TransientParamFEOpFromWeakFormWithTrian) = get_order(op)
ODEs.get_res(op::TransientParamFEOpFromWeakFormWithTrian) = get_res(op)
ODEs.get_jacs(op::TransientParamFEOpFromWeakFormWithTrian) = get_jacs(op)
ODEs.get_assembler(op::TransientParamFEOpFromWeakFormWithTrian) = get_assembler(op)
realization(op::TransientParamFEOpFromWeakFormWithTrian;kwargs...) = realization(op;kwargs...)
get_induced_norm(op::TransientParamFEOpFromWeakFormWithTrian) = get_induced_norm(op)

function assemble_norm_matrix(op::TransientParamFEOpFromWeakFormWithTrian)
  assemble_norm_matrix(op.op)
end

function ODEs.get_assembler(op::TransientParamFEOpFromWeakFormWithTrian,r::TransientParamRealization)
  get_assembler(op.op,r)
end

function set_triangulation(res,jacs::NTuple{1,Function},trian_res,trian_jacs,order)
  jac, = jacs
  trian_jac, = trian_jacs
  degree = 2*order

  meas_res = Measure.(trian_res,degree)
  meas_jac = Measure.(trian_jac,degree)

  newres(μ,t,u,v,args...) = res(μ,t,u,v,args...)
  newjac(μ,t,u,du,v,args...) = jac(μ,t,u,du,v,args...)

  newres(μ,t,u,v) = newres(μ,t,u,v,meas_res...)
  newjac(μ,t,u,du,v) = newjac(μ,t,u,du,v,meas_jac...)

  return newres,newjac
end

function set_triangulation(res,jacs::NTuple{2,Function},trian_res,trian_jacs,order)
  jac,jac_t = jacs
  trian_jac,trian_jac_t = trian_jacs
  degree = 2*order

  meas_res = Measure.(trian_res,degree)
  meas_jac = Measure.(trian_jac,degree)
  meas_jac_t = Measure.(trian_jac_t,degree)

  newres(μ,t,u,v,args...) = res(μ,t,u,v,args...)
  newjac(μ,t,u,du,v,args...) = jac(μ,t,u,du,v,args...)
  newjac_t(μ,t,u,dut,v,args...) = jac_t(μ,t,u,dut,v,args...)

  newres(μ,t,u,v) = newres(μ,t,u,v,meas_res...)
  newjac(μ,t,u,du,v) = newjac(μ,t,u,du,v,meas_jac...)
  newjac_t(μ,t,u,dut,v) = newjac_t(μ,t,u,dut,v,meas_jac_t...)

  return newres,newjac,newjac_t
end

struct TransientParamSaddlePointFEOpWithTrian{T} <: TransientParamFEOperatorWithTrian{T}
  op::TransientParamSaddlePointFEOp{T}
  trian_res::Tuple{Vararg{Triangulation}}
  trian_jacs::NTuple{2,Tuple{Vararg{Triangulation}}}
end

function TransientFEOperatorWithTrian(op::TransientParamSaddlePointFEOp,trian_res,trian_jacs...)
  newop = set_triangulation(op,trian_res,trian_jacs)
  TransientParamSaddlePointFEOpWithTrian(newop,trian_res,trian_jacs)
end

FESpaces.get_test(op::TransientParamSaddlePointFEOpWithTrian) = get_test(op)
FESpaces.get_trial(op::TransientParamSaddlePointFEOpWithTrian) = get_trial(op)
Polynomials.get_order(op::TransientParamSaddlePointFEOpWithTrian) = get_order(op)
ODEs.get_res(op::TransientParamSaddlePointFEOpWithTrian) = get_res(op)
ODEs.get_jacs(op::TransientParamSaddlePointFEOpWithTrian) = get_jacs(op)
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

# needed to retrieve measures
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
  newres,newjacs... = set_triangulation(op.res,op.jacs,trian_res,trian_jacs,polyn_order)
  TransientParamFEOperator(
    newres,newjacs,op.induced_norm,op.tpspace,op.trial,op.test)
end

function set_triangulation(op::TransientParamLinearFEOpFromWeakForm,trian_res,trian_jacs)
  polyn_order = get_polynomial_order(op.test)
  newres,newforms... = set_triangulation(op.res,op.forms,trian_res,trian_jacs,polyn_order)
  TransientParamLinearFEOperator(
    newforms,newres,op.induced_norm,op.tpspace,op.trial,op.test;constant_forms=op.constant_forms)
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
