abstract type ParamFEOperator{C<:OperatorType} <: GridapType end

"""
Parametric FE operator that is defined by a parametric weak form
"""
mutable struct ParamFEOperatorFromWeakForm{C<:OperatorType} <: ParamFEOperator{C}
  res::Function
  jac::Function
  assem::Assembler
  pspace::ParamSpace
  trial::Any
  test::FESpace
end

function ParamAffineFEOperator(res::Function,jac::Function,pspace,trial,test)
  # res(μ,u,v) = a(μ,u,v) - b(μ,v)
  # jac(μ,u,du,v) = a(μ,du,v)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOperatorFromWeakForm{Affine}(res,jac,assem,pspace,trial,test)
end

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOperatorFromWeakForm{Nonlinear}(res,jac,assem,pspace,trial,test)
end

function Gridap.FESpaces.SparseMatrixAssembler(
  trial::Union{ParamTrialFESpace,ParamMultiFieldTrialFESpace},
  test::FESpace)
  SparseMatrixAssembler(trial(nothing),test)
end

get_test(op::ParamFEOperatorFromWeakForm) = op.test

get_trial(op::ParamFEOperatorFromWeakForm) = op.trial

get_pspace(op::ParamFEOperatorFromWeakForm) = op.pspace

realization(op::ParamFEOperator,args...) = realization(op.pspace,args...)

function allocate_cache(op::ParamFEOperator)
  Us = op.trial
  U = allocate_trial_space(Us)
  U,Us
end

function update_cache!(cache,::ParamFEOperator,μ::AbstractVector)
  U,Us = cache
  evaluate!(U,Us,μ)
  U,Us
end

function _allocate_matrix_and_vector(op::ParamFEOperator,xh::AbstractVector)
  b = allocate_residual(op,xh)
  A = allocate_jacobian(op,xh)
  A,b
end

function _matrix!(
  A::AbstractMatrix,
  op::ParamFEOperator,
  xh::AbstractVector,
  μ::AbstractVector)

  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobian!(A,op,μ,xh)
end

function _vector!(
  b::AbstractVector,
  op::ParamFEOperator,
  xh::AbstractVector,
  μ::AbstractVector)

  residual!(b,op,μ,xh)
  b .*= -1.0
end

function allocate_residual(
  op::ParamFEOperatorFromWeakForm,
  xh::AbstractVector)

  test = get_test(op)
  v = get_fe_basis(test)
  trial = get_trial(op)(nothing)
  x = EvaluationFunction(trial,xh)
  vecdata = collect_cell_vector(test,op.res(realization(op),x,v))
  allocate_vector(op.assem,vecdata)
end

function allocate_jacobian(
  op::ParamFEOperatorFromWeakForm,
  xh::AbstractVector)

  test = get_test(op)
  v = get_fe_basis(test)
  trial = get_trial(op)(nothing)
  x = EvaluationFunction(trial,xh)
  du = get_trial_fe_basis(trial)
  matdata = collect_cell_matrix(trial,test,op.jac(realization(op),x,du,v))
  allocate_matrix(op.assem,matdata)
end

function residual!(
  b::AbstractVector,
  op::ParamFEOperatorFromWeakForm,
  μ::AbstractVector,
  xh::AbstractVector)

  test = get_test(op)
  v = get_fe_basis(test)
  trial = get_trial(op)(μ)
  x = EvaluationFunction(trial,xh)
  vecdata = collect_cell_vector(V,op.res(μ,x,v))
  assemble_vector!(b,op.assem,vecdata)
  b
end

function jacobian!(
  A::AbstractMatrix,
  op::ParamFEOperatorFromWeakForm,
  μ::AbstractVector,
  xh::AbstractVector)

  test = get_test(op)
  v = get_fe_basis(test)
  trial = get_trial(op)(μ)
  x = EvaluationFunction(trial,xh)
  du = get_trial_fe_basis(trial)
  matdata = collect_cell_matrix(U,V,op.jac(μ,x,du,v))
  assemble_matrix_add!(A,op.assem,matdata)
  A
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

function _collect_trian_res(op::ParamFEOperator)
  μ = realization(op)
  uh = zero(op.test)
  v = get_fe_basis(op.test)
  veccontrib = op.res(μ,uh,v)
  collect_trian(veccontrib)
end

function _collect_trian_jac(op::ParamFEOperator)
  μ = realization(op)
  uh = zero(op.test)
  v = get_fe_basis(op.test)
  matcontrib = op.jac(μ,uh,v,v)
  collect_trian(matcontrib)
end

function get_single_field(
  op::ParamFEOperator{C},
  filter::Tuple{Vararg{Int}}) where C

  r_filter,c_filter = filter
  trial = op.trial
  test = op.test
  c_trial = trial[c_filter]
  r_test = test[r_filter]
  rc_assem = SparseMatrixAssembler(c_trial,r_test)
  ParamFEOperatorFromWeakForm{C}(
    op.res,
    op.jac,
    rc_assem,
    op.pspace,
    c_trial,
    r_test)
end

# function _vecdata_residual(
#   op::ParamFEOperator,
#   ::FESolver,
#   sols::AbstractArray,
#   params::AbstractArray,
#   filter::Tuple{Vararg{Int}},
#   args...;
#   trian::Triangulation=get_triangulation(op.test))

#   row,_ = filter
#   test_row = get_test(op)[row]
#   trial = get_trial(op)
#   dv_row = _get_fe_basis(op.test,row)
#   sol_μ = _as_param_function(sols,params)
#   assem_row = SparseMatrixAssembler(test_row,test_row)
#   op.assem = assem_row

#   function vecdata(μ)
#     u = EvaluationFunction(trial(μ),sol_μ(μ))
#     collect_cell_vector(test_row,op.res(μ,u,dv_row,args...),trian)
#   end

#   vecdata
# end

# function _matdata_jacobian(
#   op::ParamFEOperator,
#   ::FESolver,
#   sols::AbstractArray,
#   params::AbstractArray,
#   filter::Tuple{Vararg{Int}},
#   args...;
#   trian::Triangulation=get_triangulation(op.test))

#   row,col = filter
#   test_row = get_test(op)[row]
#   trial_col = get_trial(op)[col]
#   dv_row = _get_fe_basis(op.test,row)
#   du_col = _get_trial_fe_basis(op.trial(nothing),col)
#   sols_col = isa(sols,AbstractMatrix) ? sols : sols[col]
#   sol_col_μ = _as_param_function(sols_col,params)
#   assem_row_col = SparseMatrixAssembler(trial_col(nothing)[col],test_row)
#   op.assem = assem_row_col

#   function matdata(μ)
#     u_col = EvaluationFunction(trial_col(μ),sol_col_μ(μ))
#     collect_cell_matrix(trial_col(μ),test_row,op.jac(μ,u_col(μ),dv_row,du_col,args...),trian)
#   end

#   matdata
# end
