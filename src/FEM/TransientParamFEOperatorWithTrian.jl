# interface to accommodate the separation of terms depending on the triangulation
struct TransientParamFEOperatorWithTrian{T<:OperatorType,A,B,C} <: TransientParamFEOperator{T}
  op::A
  trian_res::B
  trian_jacs::C
  function TransientParamFEOperatorWithTrian(
    op::TransientParamFEOperator{T},
    trian_res::B,
    trian_jacs::C) where {T,B,C}

    A = typeof(op)
    new{T,A,B,C}(op,trian_res,trian_jacs)
  end
end

function FEOperatorWithTrian(op::TransientParamFEOperator,trian_res,trian_jacs...)
  newop = set_triangulation(op,trian_res,trian_jacs)
  TransientParamFEOperatorWithTrian(newop,trian_res,trian_jacs)
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

const TransientParamSaddlePointFEOperatorWithTrian = TransientParamFEOperatorWithTrian{T,A
  } where {T<:OperatorType,A<:TransientParamSaddlePointFEOperator{T}}

function assemble_coupling_matrix(op::TransientParamSaddlePointFEOperatorWithTrian)
  assemble_coupling_matrix(op.op)
end

function _remove_saddle_point_operator(op::TransientParamSaddlePointFEOperatorWithTrian)
  TransientParamFEOperatorWithTrian(op.op.op,op.trian_res,op.trian_jacs)
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

function set_triangulation(op::TransientParamFEOperatorFromWeakForm{T},trian_res,trian_jacs) where T
  polyn_order = get_polynomial_order(op.test)
  newres,newjacs... = set_triangulation(op.res,op.jacs,trian_res,trian_jacs,polyn_order)
  TransientParamFEOperatorFromWeakForm{T}(
    newres,op.rhs,newjacs,op.induced_norm,op.assem,
    op.tpspace,op.trials,op.test,op.order)
end

function set_triangulation(op::TransientParamSaddlePointFEOperator,trian_res,trian_jacs)
  newop = set_triangulation(op.op,trian_res,trian_jacs)
  TransientParamSaddlePointFEOperator(newop,op.coupling)
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
  newop = set_triangulation(op,newtrian_res,newtrian_jacs...)
  TransientParamFEOperatorWithTrian(newop,args...)
end

function Algebra.allocate_residual(
  op::TransientParamSaddlePointFEOperatorWithTrian,
  r::TransientParamRealization,
  uh::T,
  cache) where T

  newop = _remove_saddle_point_operator(op)
  allocate_residual(newop,r,uh,cache)
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
  op::TransientParamSaddlePointFEOperatorWithTrian,
  r::TransientParamRealization,
  uh::T,
  cache) where T

  newop = _remove_saddle_point_operator(op)
  _allocate_jacobian(newop,r,uh,cache)
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
  op::TransientParamSaddlePointFEOperatorWithTrian,
  r::TransientParamRealization,
  xh::T,
  cache) where T

  newop = _remove_saddle_point_operator(op)
  residual!(b,newop,r,xh,cache)
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
  op::TransientParamSaddlePointFEOperatorWithTrian,
  r::TransientParamRealization,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache) where T

  newop = _remove_saddle_point_operator(op)
  jacobian!(A,newop,r,xh,i,γᵢ,cache)
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
  A::Contribution,
  op::TransientParamSaddlePointFEOperatorWithTrian,
  r::TransientParamRealization,
  xh::T,
  γ::Tuple{Vararg{Real}},
  cache) where T

  newop = _remove_saddle_point_operator(op)
  jacobians!(A,newop,r,xh,γ,cache)
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
