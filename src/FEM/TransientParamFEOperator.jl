"""
A parametric version of the `Gridap` `TransientFEOperator`
"""
abstract type TransientParamFEOperator{T<:OperatorType} <: GridapType end

function FESpaces.get_algebraic_operator(feop::TransientParamFEOperator{C}) where C
  ODEParamOpFromFEOp{C}(feop)
end

function TransientFETools.allocate_cache(op::TransientParamFEOperator)
  nothing
end

function TransientFETools.update_cache!(cache::Nothing,op::TransientParamFEOperator,r)
  nothing
end

"""
Transient FE operator that is defined by a transient Weak form
"""
struct TransientParamFEOperatorFromWeakForm{T<:OperatorType} <: TransientParamFEOperator{T}
  res::Function
  rhs::Function
  jacs::Tuple{Vararg{Function}}
  assem::Assembler
  tpspace::TransientParamSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function AffineTransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOperatorFromWeakForm{Affine}(
    res,TransientFETools.rhs_error,(jac,jac_t),assem,tpspace,(trial,∂t(trial)),test,1)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOperatorFromWeakForm{Nonlinear}(
    res,TransientFETools.rhs_error,(jac,jac_t),assem,tpspace,(trial,∂t(trial)),test,1)
end

FESpaces.get_test(op::TransientParamFEOperatorFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamFEOperatorFromWeakForm) = op.trials[1]
ReferenceFEs.get_order(op::TransientParamFEOperatorFromWeakForm) = op.order
realization(op::TransientParamFEOperatorFromWeakForm;kwargs...) = realization(op.tpspace;kwargs...)

function FESpaces.SparseMatrixAssembler(
  trial::Union{TransientTrialParamFESpace,TransientMultiFieldTrialParamFESpace},
  test::FESpace)
  SparseMatrixAssembler(trial(nothing),test)
end

function TransientFETools.rhs_error(μ,t,u,v)
  error("The \"rhs\" function is not defined for this TransientFEOperator.
  Please, try to use another type of TransientFEOperator that supports this
  functionality.")
end

function Algebra.allocate_residual(
  op::TransientParamFEOperatorFromWeakForm,
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
  dc = op.res(get_params(r),get_times(r),xh,v)
  assem = get_param_assembler(op.assem,r)
  vecdata = collect_cell_vector(test,dc)
  allocate_vector(assem,vecdata)
end

function Algebra.allocate_jacobian(
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  uh::CellField,
  cache)

  _matdata_jacobians = TransientFETools.fill_initial_jacobians(op,r,uh)
  matdata = TransientFETools._vcat_matdata(_matdata_jacobians)
  assem = get_param_assembler(op.assem,r)
  allocate_matrix(assem,matdata)
end

function Algebra.residual!(
  b::AbstractVector,
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  cache) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dc = op.res(get_params(r),get_times(r),xh,v)
  vecdata = collect_cell_vector(test,dc)
  assem = get_param_assembler(op.assem,r)
  assemble_vector!(b,assem,vecdata)
  b
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache) where T

  matdata = _matdata_jacobian(op,r,xh,i,γᵢ)
  assem = get_param_assembler(op.assem,r)
  assemble_matrix_add!(A,assem,matdata)
  A
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  γ::Tuple{Vararg{Real}},
  cache) where T

  _matdata_jacobians = TransientFETools.fill_jacobians(op,r,xh,γ)
  matdata = TransientFETools._vcat_matdata(_matdata_jacobians)
  assem = get_param_assembler(op.assem,r)
  assemble_matrix_add!(A,assem,matdata)
  A
end

function TransientFETools.fill_initial_jacobians(
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  uh::T) where T

  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  _matdata = ()
  for i in 1:get_order(op)+1
    _matdata = (_matdata...,_matdata_jacobian(op,r,xh,i,0.0))
  end
  return _matdata
end

function TransientFETools.fill_jacobians(
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  γ::Tuple{Vararg{Real}}) where T

  _matdata = ()
  for i in 1:get_order(op)+1
    if (γ[i] > 0.0)
      _matdata = (_matdata...,_matdata_jacobian(op,r,xh,i,γ[i]))
    end
  end
  return _matdata
end

function TransientFETools._matdata_jacobian(
  op::TransientParamFEOperatorFromWeakForm,
  r::TransientParamRealization,
  xh::T,
  i::Integer,
  γᵢ::Real) where T

  trial = evaluate(get_trial(op),nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = γᵢ*op.jacs[i](get_params(r),get_times(r),xh,u,v)
  collect_cell_matrix(trial,test,dc)
end

# interface to accommodate the separation of terms depending on the triangulation
struct TransientParamFEOperatorWithTrian{T<:OperatorType,A,B} <: TransientParamFEOperator{T}
  op::TransientParamFEOperatorFromWeakForm{T}
  trian_res::A
  trian_jacs::B
end

function set_triangulation(res,jac,jac_t,trian_res,trian_jac,trian_jac_t,order)
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

function AffineTransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test,trian_res,trian_jac,trian_jac_t)

  order = get_polynomial_order(test)
  newres,newjac,newjac_t = set_triangulation(res,jac,jac_t,trian_res,trian_jac,trian_jac_t,order)
  op = AffineTransientParamFEOperator(newres,newjac,newjac_t,tpspace,trial,test)
  TransientParamFEOperatorWithTrian(op,trian_res,(trian_jac,trian_jac_t))
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test,trian_res,trian_jac,trian_jac_t)

  order = get_polynomial_order(test)
  newres,newjac,newjac_t = set_triangulation(res,jac,jac_t,trian_res,trian_jac,trian_jac_t,order)
  op = TransientParamFEOperator(newres,newjac,newjac_t,tpspace,trial,test)
  TransientParamFEOperatorWithTrian(op,trian_res,(trian_jac,trian_jac_t))
end

FESpaces.get_test(op::TransientParamFEOperatorWithTrian) = get_test(op.op)
FESpaces.get_trial(op::TransientParamFEOperatorWithTrian) = get_trial(op.op)
ReferenceFEs.get_order(op::TransientParamFEOperatorWithTrian) = get_order(op.op)
realization(op::TransientParamFEOperatorWithTrian;kwargs...) = realization(op.op;kwargs...)

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
  newtrian_jac,newtrian_jac_t = order_triangulations.(op.trian_jacs,trian_jacs)
  @unpack res,rhs,jacs,assem,tpspace,trials,test,order = op.op
  jac,jac_t = jacs
  porder = get_polynomial_order(test)
  newres,newjacs... = set_triangulation(res,jac,jac_t,newtrian_res,newtrian_jac,newtrian_jac_t,porder)
  feop = TransientParamFEOperatorFromWeakForm{T}(newres,rhs,newjacs,assem,tpspace,trials,test,order)
  TransientParamFEOperatorWithTrian(feop,newtrian_res,(newtrian_jac,newtrian_jac_t))
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
    Ai = A[i]
    dc = op.op.jacs[i](get_params(r),get_times(r),xh,u,v)
    Ai = contribution(op.trian_res) do trian
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
  assem = get_param_assembler(op.assem,r)
  dc = γᵢ*op.jacs[i](get_params(r),get_times(r),xh,u,v)
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

# interface to accommodate the separation of terms depending on the linearity/nonlinearity
struct LinearNonlinearTransientParamFEOperator{T<:OperatorType} <: TransientParamFEOperator{T}
  op_linear::TransientParamFEOperator{T}
  op_nonlinear::TransientParamFEOperator{T}
end

function TransientFETools.test_transient_fe_operator(op::TransientParamFEOperator,uh,μt)
  odeop = get_algebraic_operator(op)
  @test isa(odeop,ODEParamOperator)
  cache = allocate_cache(op)
  V = get_test(op)
  @test isa(V,FESpace)
  U = get_trial(op)
  U0 = U(μt)
  @test isa(U0,TrialParamFESpace)
  r = allocate_residual(op,μt,uh,cache)
  @test isa(r,ParamVector)
  xh = TransientCellField(uh,(uh,))
  residual!(r,op,μt,xh,cache)
  @test isa(r,ParamVector)
  J = allocate_jacobian(op,μt,uh,cache)
  @test isa(J,ParamMatrix)
  jacobian!(J,op,μt,xh,1,1.0,cache)
  @test isa(J,ParamMatrix)
  jacobian!(J,op,μt,xh,2,1.0,cache)
  @test isa(J,ParamMatrix)
  jacobians!(J,op,μt,xh,(1.0,1.0),cache)
  @test isa(J,ParamMatrix)
  cache = update_cache!(cache,op,μt)
  true
end
