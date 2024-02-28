function AffineTransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,
  tpspace,trial,test,trian_res,trian_jac)

  jacs = (jac,)
  trian_jacs = (trian_jac,)
  order = get_polynomial_order(test)
  newres,newjac = set_triangulation(res,jacs,trian_res,trian_jacs,order)
  op = AffineTransientParamFEOperator(newres,newjac,induced_norm,tpspace,trial,test)
  TransientParamFEOperatorWithTrian(op,trian_res,trian_jacs)
end

function AffineTransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,
  tpspace,trial,test,trian_res,trian_jac,trian_jac_t)

  jacs = (jac,jac_t)
  trian_jacs = (trian_jac,trian_jac_t)
  order = get_polynomial_order(test)
  newres,newjac,newjac_t = set_triangulation(res,jacs,trian_res,trian_jacs,order)
  op = AffineTransientParamFEOperator(newres,newjac,newjac_t,induced_norm,tpspace,trial,test)
  TransientParamFEOperatorWithTrian(op,trian_res,trian_jacs)
end

function TransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,
  tpspace,trial,test,trian_res,trian_jac)

  jacs = (jac,)
  trian_jacs = (trian_jac,)
  order = get_polynomial_order(test)
  newres,newjac = set_triangulation(res,jacs,trian_res,trian_jacs,order)
  op = TransientParamFEOperator(newres,newjac,induced_norm,tpspace,trial,test)
  TransientParamFEOperatorWithTrian(op,trian_res,trian_jacs)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,
  tpspace,trial,test,trian_res,trian_jac,trian_jac_t)

  jacs = (jac,jac_t)
  trian_jacs = (trian_jac,trian_jac_t)
  order = get_polynomial_order(test)
  newres,newjac,newjac_t = set_triangulation(res,jacs,trian_res,trian_jacs,order)
  op = TransientParamFEOperator(newres,newjac,newjac_t,induced_norm,tpspace,trial,test)
  TransientParamFEOperatorWithTrian(op,trian_res,trian_jacs)
end

# interface to accommodate the separation of terms depending on the triangulation
struct TransientParamFEOperatorWithTrian{T<:OperatorType,A} <: TransientParamFEOperator{T}
  op::A
  trian_res::Triangulation
  trian_jacs::Tuple{Vararg{Triangulation}}
  function TransientParamFEOperatorWithTrian(
    op::TransientParamFEOperatorFromWeakForm{T},
    trian_res::Triangulation,
    trian_jacs::Tuple{Vararg{Triangulation}}) where T

    A = typeof(op)
    new{T,A}(op,trian_res,trian_jacs)
  end
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

FESpaces.get_test(op::TransientParamFEOperatorWithTrian) = get_test(op.op)
FESpaces.get_trial(op::TransientParamFEOperatorWithTrian) = get_trial(op.op)
ReferenceFEs.get_order(op::TransientParamFEOperatorWithTrian) = get_order(op.op)
realization(op::TransientParamFEOperatorWithTrian;kwargs...) = realization(op.op;kwargs...)

function assemble_norm_matrix(op::TransientParamFEOperatorWithTrian)
  assemble_norm_matrix(op.op)
end

const TransientParamSaddlePointFEOperatorWithTrian = TransientParamFEOperatorWithTrian{
  T<:OperatorType,A} where {T,A<:TransientParamSaddlePointFEOperator}

function compute_coupling_matrix(op::TransientParamSaddlePointFEOperatorWithTrian)
  compute_coupling_matrix(op.op)
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

# change triangulation
function change_triangulation(
  op::TransientParamFEOperatorWithTrian{T},
  trian_jacs,
  trian_res) where T

  newtrian_res = order_triangulations(op.trian_res,trian_res)
  newtrian_jacs = order_triangulations.(op.trian_jacs,trian_jacs)
  @unpack res,rhs,jacs,assem,tpspace,trials,test,order = op.op

  porder = get_polynomial_order(test)
  newres,newjacs... = set_triangulation(res,jacs,newtrian_res,newtrian_jacs,porder)
  feop = TransientParamFEOperatorFromWeakForm{T}(newres,rhs,newjacs,assem,tpspace,trials,test,order)
  TransientParamFEOperatorWithTrian(feop,newtrian_res,newtrian_jacs)
end

function Algebra.allocate_residual(
  op::TransientParamFEOperatorWithTrian,
  r::TransientParamRealization,
  uh::T,
  cache) where T

  _allocate_residual(op,r,uh,cache)
end

function _allocate_residual(
  op::TransientParamFEOperatorWithTrian,
  r::TransientParamRealization,
  uh::T,
  cache) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  dc = op.op.res(get_params(r),get_times(r),xh,v)
  assem = get_param_assembler(op.op.assem,r)
  b = contribution(op.trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end
  b
end

function Algebra.allocate_jacobian(
  op::TransientParamFEOperatorWithTrian,
  r::TransientParamRealization,
  uh::T,
  cache) where T

  _allocate_jacobian(op,r,uh,cache)
end

function _allocate_jacobian(
  op::TransientParamFEOperatorWithTrian,
  r::TransientParamRealization,
  uh::T,
  cache) where T

  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  trial = evaluate(get_trial(op),nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op.assem,r)
  A = ()
  for i = 1:get_order(op)+1
    dc = op.op.jacs[i](get_params(r),get_times(r),xh,u,v)
    Ai = contribution(op.trian_jacs[i]) do trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      allocate_matrix(assem,matdata)
    end
    A = (A...,Ai)
  end
  A
end

function Algebra.residual!(
  b::Contribution,
  op::TransientParamFEOperatorWithTrian,
  r::TransientParamRealization,
  xh::T,
  cache) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dc = op.op.res(get_params(r),get_times(r),xh,v)
  assem = get_param_assembler(op.op.assem,r)
  map(b.values,op.trian_res) do btrian,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector!(btrian,assem,vecdata)
  end
  b
end

function Algebra.jacobian!(
  A::Contribution,
  op::TransientParamFEOperatorWithTrian,
  r::TransientParamRealization,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache) where T

  trial = evaluate(get_trial(op),nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op.assem,r)
  dc = γᵢ*op.op.jacs[i](get_params(r),get_times(r),xh,u,v)
  map(A.values,op.trian_jacs[i]) do Atrian,trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    assemble_matrix_add!(Atrian,assem,matdata)
  end
  A
end

function ODETools.jacobians!(
  A::Tuple{Vararg{Contribution}},
  op::TransientParamFEOperatorWithTrian,
  r::TransientParamRealization,
  xh::T,
  γ::Tuple{Vararg{Real}},
  cache) where T

  trial = evaluate(get_trial(op),nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op.assem,r)

  for i = 1:get_order(op)+1
    Ai = A[i]
    dc = γ[i]*op.op.jacs[i](get_params(r),get_times(r),xh,u,v)
    map(Ai.values,op.trian_jacs[i]) do Atrian,trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      assemble_matrix_add!(Atrian,assem,matdata)
    end
  end
  A
end
