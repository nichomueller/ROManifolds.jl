abstract type ParamOp{C} <: GridapType end
const AffineParamOp = ParamOp{Affine}

"""
A wrapper of `ParamFEOperator` that transforms it to `ParamOp`, i.e.,
takes A(μ,uh,vh) and returns A(μ,uF), where uF represents the free values
of the `EvaluationFunction` uh
"""
struct ParamOpFromFEOp{C} <: ParamOp{C}
  feop::ParamFEOperator{C}
end

function Gridap.ODEs.TransientFETools.allocate_residual(op::ParamOpFromFEOp,uh)
  allocate_residual(op.feop,uh)
end

function Gridap.ODEs.TransientFETools.allocate_jacobian(op::ParamOpFromFEOp,uh)
  allocate_jacobian(op.feop,uh)
end

# Affine

function _allocate_matrix_and_vector(op::ParamOpFromFEOp,uh)
  b = allocate_residual(op,uh)
  A = allocate_jacobian(op,uh)
  A,b
end

function _matrix!(
  A::AbstractMatrix,
  op::ParamOpFromFEOp,
  uh,
  μ::AbstractVector)

  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobian!(A,op.feop,μ,uh)
end

function _vector!(
  b::AbstractVector,
  op::ParamOpFromFEOp,
  uh,
  μ::AbstractVector)

  residual!(b,op.feop,μ,uh)
  b .*= -1.0
end

# Nonlinear

struct ParamNonlinearOperator{T} <: NonlinearOperator
  param_op::ParamOp
  uh::T
  μ::AbstractVector
  cache
end

function Gridap.ODEs.TransientFETools.residual!(
  b::AbstractVector,
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  residual!(b,feop,op.μ,u)
end

function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobian!(A,feop,op.μ,u)
end

function Gridap.ODEs.TransientFETools.allocate_residual(
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  allocate_residual(feop,u)
end

function Gridap.ODEs.TransientFETools.allocate_jacobian(
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  allocate_jacobian(feop,u)
end

# MDEIM snapshots generation interface

_get_fe_basis(test,args...) = get_fe_basis(test)

_get_trial_fe_basis(trial,args...) = get_trial_fe_basis(trial)

function _get_fe_basis(
  test::MultiFieldFESpace,
  filter::Tuple{Vararg{Int}})

  row, = filter
  nfields = length(test.spaces)
  dv_row = Vector{Nothing}(undef,nfields)
  dv_row[row] = get_fe_basis(test[row])
  dv_row
end

function _get_trial_fe_basis(
  trial::MultiFieldFESpace,
  filter::Tuple{Vararg{Int}})

  row,col = filter
  nfields = length(du)
  du_col = Vector{Nothing}(undef,nfields)
  du_col[col] = get_trial_fe_basis(trial[col])
  du_col
end

function _vecdata_residual(
  op::ParamFEOperator,
  ::FESolver,
  sols::AbstractArray,
  params::AbstractArray,
  filter::Tuple{Vararg{Int}},
  args...;
  kwargs...)

  row,_ = filter
  trial = get_trial(op)
  dv_row = _get_fe_basis(op.test,row)
  sol_μ = _as_function(sols,params)

  function vecdata(μ)
    u = EvaluationFunction(trial(μ),sol_μ(μ))
    collect_cell_vector(test,op.res(μ,u,dv_row),args...)
  end

  μ -> _filter_vecdata(op.assem,vecdata(μ),filter)
end

function Gridap.ODEs.TransientFETools._matdata_jacobian(
  op::ParamFEOperator,
  ::FESolver,
  sols::AbstractArray,
  params::AbstractArray,
  filter::Tuple{Vararg{Int}},
  args...;
  kwargs...)

  row,col = filter
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(get_trial(op)(nothing,nothing),col)
  test_row = get_test(op)[row]
  trial_col = get_trial(op)[col]
  sols_col = sols[col]
  sol_col_μ = _as_function(sols_col,params)

  function matdata(μ)
    u_col = EvaluationFunction(trial_col(μ),sol_col_μ(μ))
    collect_cell_matrix(trial_col(μ),test_row,op.jac(μ,u_col(μ),dv_row,du_col),args...)
  end

  μ -> _filter_matdata(op.assem,matdata(μ),filter)
end
