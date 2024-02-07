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
  m::Function,a::Function,b::Function,tpspace,trial,test)
  res(μ,t,u,v) = m(μ,t,∂t(u),v) + a(μ,t,u,v) - b(μ,t,v)
  rhs(μ,t,u,v) = b(μ,t,v) - a(μ,t,u,v)
  jac(μ,t,u,du,v) = a(μ,t,du,v)
  jac_t(μ,t,u,dut,v) = m(μ,t,dut,v)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOperatorFromWeakForm{Affine}(
    res,rhs,(jac,jac_t),assem,tpspace,(trial,∂t(trial)),test,1)
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
get_fe_operator(op::TransientParamFEOperatorFromWeakForm) = op

function FESpaces.SparseMatrixAssembler(
  trial::Union{TransientTrialParamFESpace,TransientMultiFieldTrialParamFESpace},
  test::FESpace)
  SparseMatrixAssembler(trial(nothing),test)
end

function TransientFETools.rhs_error(μ,t,xh,v)
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

function AffineTransientParamFEOperator(
  m::Function,a::Function,b::Function,tpspace,trial,test,trian_m,trian_a,trian_b)

  order = get_polynomial_order(test)
  meas_m = Measure.(trian_m,order)
  meas_a = Measure.(trian_a,order)
  meas_b = Measure.(trian_b,order)

  newm(μ,t,dut,v,args...) = m(μ,t,dut,v,args...)
  newa(μ,t,du,v,args...) = a(μ,t,du,v,args...)
  newb(μ,t,v,args...) = b(μ,t,v,args...)

  newm(μ,t,dut,v) = newm(μ,t,dut,v,meas_m...)
  newa(μ,t,du,v) = newa(μ,t,du,v,meas_a...)
  newb(μ,t,v) = newb(μ,t,v,meas_b...)

  op = AffineTransientParamFEOperator(newm,newa,newb,tpspace,trial,test)
  TransientParamFEOperatorWithTrian(op,trian_b,(trian_a,trian_m))
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test,trian_res,trian_jac,trian_jac_t)

  order = get_polynomial_order(test)
  meas_res = Measure.(trian_res,order)
  meas_jac = Measure.(trian_jac,order)
  meas_jac_t = Measure.(trian_jac_t,order)

  newres(μ,t,u,v,args...) = res(μ,t,u,v,args...)
  newjac(μ,t,u,du,v,args...) = jac(μ,t,u,du,v)
  newjac_t(μ,t,u,dut,v,args...) = jac_t(μ,t,u,dut,v,args...)

  newres(μ,t,u,v,args...) = newres(μ,t,u,v,meas_res...)
  newjac(μ,t,u,du,v,args...) = newjac(μ,t,u,du,v,meas_jac...)
  newjac_t(μ,t,u,dut,v,args...) = newjac_t(μ,t,u,dut,v,meas_jac_t...)

  op = TransientParamFEOperator(res,jac,jac_t,tpspace,trial,test)
  TransientParamFEOperatorWithTrian(op,trian_res,(trian_jac,trian_jac_t))
end

FESpaces.get_test(op::TransientParamFEOperatorWithTrian) = get_test(op.op)
FESpaces.get_trial(op::TransientParamFEOperatorWithTrian) = get_trial(op.op)
ReferenceFEs.get_order(op::TransientParamFEOperatorWithTrian) = get_order(op.op)
realization(op::TransientParamFEOperatorWithTrian;kwargs...) = realization(op.op;kwargs...)
get_fe_operator(op::TransientParamFEOperatorWithTrian) = op.op

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

get_polynomial_order(fs::SingleFieldParamFESpace) = get_polynomial_order(get_fe_basis(fs),get_background_model(fs))
get_polynomial_order(fs::MultiFieldFESpace) = maximum(map(get_polynomial_order,fs.spaces))

# change triangulation
function change_triangulation(
  op::TransientParamFEOperatorWithTrian{Affine},
  newtrian_b,newtrian_a,newtrian_m)

  @unpack res,jac,jac_t,tpspace,trial,test,trian_res,trian_jac = op
  AffineTransientParamFEOperator(res,jac,jac_t,tpspace,trial,test,newtrian_m,newtrian_a,newtrian_b)
end

function change_triangulation(
  op::TransientParamFEOperatorWithTrian,
  newtrian_res,newtrian_jac,newtrian_jac_t)

  @unpack res,jac,jac_t,tpspace,trial,test,trian_res,trian_jac = op
  TransientParamFEOperator(res,jac,jac_t,tpspace,trial,test,newtrian_res,newtrian_jac,newtrian_jac_t)
end

function Algebra.allocate_residual(
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
  b = array_contribution()
  for trian in op.trian_res
    vecdata = collect_cell_vector(test,dc)
    b[trian] = allocate_vector(assem,vecdata)
  end
  b
end

function Algebra.allocate_jacobian(
  op::TransientParamFEOperatorWithTrian,
  r::TransientParamRealization,
  uh::CellField,
  cache)

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
    Ai = array_contribution()
    dc = op.op.jacs[i](get_params(r),get_times(r),xh,u,v)
    for trian in op.trian_jacs[i]
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      Ai[trian] = allocate_matrix(assem,matdata)
    end
    A = (A...,Ai)
  end
  A
end

function Algebra.residual!(
  b::ArrayContribution,
  op::TransientParamFEOperatorWithTrian,
  r::TransientParamRealization,
  xh::T,
  cache) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dc = op.op.res(get_params(r),get_times(r),xh,v)
  assem = get_param_assembler(op.op.assem,r)
  for trian in op.trian_res
    btrian = b[trian]
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector!(btrian,assem,vecdata)
  end
  b
end

function Algebra.jacobian!(
  A::ArrayContribution,
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
  for trian in op.trian_jacs[i]
    Atrian = A[trian]
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    assemble_matrix_add!(Atrian,assem,matdata)
  end
  A
end

function TransientFETools.jacobians!(
  A::Tuple{Vararg{ArrayContribution}},
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
    for trian in op.trian_jacs[i]
      Atrian = Ai[trian]
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      assemble_matrix_add!(Atrian,assem,matdata)
    end
  end
  A
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
