abstract type ParamOperator{C} <: GridapType end
const AffineParamOp = ParamOperator{Affine}

"""
A wrapper of `ParamFEOperator` that transforms it to `ParamOperator`, i.e.,
takes A(μ,uh,vh) and returns A(μ,uF), where uF represents the free values
of the `EvaluationFunction` uh
"""
struct ParamOpFromFEOp{C} <: ParamOperator{C}
  feop::ParamFEOperator{C}
end

function allocate_residual(op::ParamOpFromFEOp,uh)
  allocate_residual(op.feop,uh)
end

function allocate_jacobian(op::ParamOpFromFEOp,uh)
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
  param_op::ParamOperator
  uh::T
  μ::AbstractVector
  cache
end

function residual!(
  b::AbstractVector,
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  residual!(b,feop,op.μ,u)
end

function jacobian!(
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

function allocate_residual(
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  allocate_residual(feop,u)
end

function allocate_jacobian(
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
  row::Int)

  dv_row = Any[]
  for nf = eachindex(test.spaces)
    nf == row ? push!(dv_row,get_fe_basis(test[row])) : push!(dv_row,nothing)
  end
  dv_row
end

function _get_trial_fe_basis(
  trial::MultiFieldFESpace,
  col::Int)

  du_col = Any[]
  for nf = eachindex(trial.spaces)
    nf == col ? push!(du_col,get_trial_fe_basis(trial[col])) : push!(du_col,nothing)
  end
  du_col
end

function _vecdata_residual(
  op::ParamFEOperator,
  ::FESolver,
  sols::AbstractArray,
  params::AbstractArray,
  filter::Tuple{Vararg{Int}},
  args...;
  trian::Triangulation=get_triangulation(op.test))

  row,_ = filter
  test_row = get_test(op)[row]
  trial = get_trial(op)
  dv_row = _get_fe_basis(op.test,row)
  sol_μ = _as_param_function(sols,params)
  assem_row = SparseMatrixAssembler(test_row,test_row)
  op.assem = assem_row

  function vecdata(μ)
    u = EvaluationFunction(trial(μ),sol_μ(μ))
    collect_cell_vector(test_row,op.res(μ,u,dv_row,args...),trian)
  end

  vecdata
end

function _matdata_jacobian(
  op::ParamFEOperator,
  ::FESolver,
  sols::AbstractArray,
  params::AbstractArray,
  filter::Tuple{Vararg{Int}},
  args...;
  trian::Triangulation=get_triangulation(op.test))

  row,col = filter
  test_row = get_test(op)[row]
  trial_col = get_trial(op)[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(op.trial(nothing),col)
  sols_col = isa(sols,AbstractMatrix) ? sols : sols[col]
  sol_col_μ = _as_param_function(sols_col,params)
  assem_row_col = SparseMatrixAssembler(trial_col(nothing)[col],test_row)
  op.assem = assem_row_col

  function matdata(μ)
    u_col = EvaluationFunction(trial_col(μ),sol_col_μ(μ))
    collect_cell_matrix(trial_col(μ),test_row,op.jac(μ,u_col(μ),dv_row,du_col,args...),trian)
  end

  matdata
end
