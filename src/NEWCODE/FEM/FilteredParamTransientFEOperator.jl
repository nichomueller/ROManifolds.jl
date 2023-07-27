struct FilteredParamTransientFEOperator{C<:OperatorType} <: ParamTransientFEOperator{C}
  res::Function
  jac::Function
  assem::Assembler
  trial::Any
  test::FESpace
  trial_basis::Any
  test_basis::Any
end

function allocate_cache(op::FilteredParamTransientFEOperator)
  Ut = get_trial(op)
  U = allocate_trial_space(Ut)
  Uts = (Ut,)
  Us = (U,)
  for i in 1:get_order(op)
    Uts = (Uts...,∂t(Uts[i]))
    Us = (Us...,allocate_trial_space(Uts[i+1]))
  end
  fecache = nothing
  ode_cache = (Us,Uts,fecache)
  ode_cache
end

function update_cache!(
  ode_cache,
  op::FilteredParamTransientFEOperator,
  μ::AbstractVector,
  t::Real)

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],μ,t))
  end
  fecache = nothing
  (Us,Uts,fecache)
end

function allocate_evaluation_function(op::FilteredParamTransientFEOperator)
  μ,t = realization(op),1.
  trial = get_trial(op)(μ,t)
  xh = EvaluationFunction(trial,fill(0.,num_free_dofs(op.test)))
  dxh = ()
  for _ in 1:get_order(op)
    dxh = (dxh...,xh)
  end
  TransientCellField(xh,dxh)
end

function evaluation_function(
  op::FilteredParamTransientFEOperator,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
end

filter_evaluation_function(u,args...) = u

function filter_evaluation_function(u::TransientMultiFieldCellField,col::Int)
  u_col = Any[]
  for nf = eachindex(u.transient_single_fields)
    nf == col ? push!(u_col,u[col]) : push!(u_col,nothing)
  end
  u_col
end

function allocate_residual(
  op::ParamTransientFEOperatorFromWeakForm,
  args...;
  assem::SparseMatrixAssembler=op.assem)

  xh = allocate_evaluation_function(op)
  v = get_fe_basis(op.test)
  vecdata = collect_cell_vector(op.test,op.res(realization(op),0.0,xh,v))
  allocate_vector(assem,vecdata)
end

function residual!(
  b::AbstractVector,
  op::ParamTransientFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::Real,
  xhF::Tuple{Vararg{AbstractVector}},
  cache)

  xh = evaluation_function(op,xhF,cache)
  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(μ,t,xh,v))
  assemble_vector!(b,op.assem,vecdata)
  b
end

function allocate_jacobian(
  op::ParamTransientFEOperatorFromWeakForm,
  args...;
  assem::SparseMatrixAssembler=op.assem)

  _matdata_jacobians = fill_initial_jacobians(op)
  matdata = _vcat_matdata(_matdata_jacobians)
  allocate_matrix(assem,matdata)
end

function jacobian!(
  A::AbstractMatrix,
  op::ParamTransientFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::Real,
  xhF::Tuple{Vararg{AbstractVector}},
  i::Integer,
  γᵢ::Real,
  cache)

  xh = evaluation_function(op,xhF,cache)
  matdata = _matdata_jacobian(op,μ,t,xh,i,γᵢ)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function jacobians!(
  A::AbstractMatrix,
  op::ParamTransientFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::Real,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  cache)

  xh = evaluation_function(op,xhF,cache)
  _matdata_jacobians = fill_jacobians(op,μ,t,xh,γ)
  matdata = _vcat_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function fill_initial_jacobians(op::ParamTransientFEOperatorFromWeakForm,args...)
  xh = allocate_evaluation_function(op)
  _matdata = ()
  for i in 1:get_order(op)+1
    _matdata = (_matdata...,_matdata_jacobian(op,realization(op),0.0,xh,i,0.0))
  end
  return _matdata
end

function fill_jacobians(
  op::ParamTransientFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::Real,
  uh::T,
  γ::Tuple{Vararg{Real}}) where T

  _matdata = ()
  for i in 1:get_order(op)+1
    if (γ[i] > 0.0)
      _matdata = (_matdata...,_matdata_jacobian(op,μ,t,uh,i,γ[i]))
    end
  end
  return _matdata
end

function _matdata_jacobian(
  op::ParamTransientFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::Real,
  uh::T,
  i::Integer,
  γᵢ::Real) where T

  Uh = get_trial(op)(nothing,nothing)
  V = get_test(op)
  du = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  collect_cell_matrix(Uh,V,γᵢ*op.jacs[i](μ,t,uh,du,v))
end

function _collect_trian_res(op::FilteredParamTransientFEOperator)
  μ,t = realization(op),0.
  uh = zero(op.test)
  v = get_fe_basis(op.test)
  dxh = ()
  for _ in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  veccontrib = op.res(μ,t,xh,v)
  collect_trian(veccontrib)
end

function _collect_trian_jac(op::FilteredParamTransientFEOperator)
  μ,t = realization(op),0.
  uh = zero(op.test)
  v = get_fe_basis(op.test)
  trians = ()
  for j in op.jacs
    matcontrib = j(μ,t,uh,v,v)
    trians = (trians...,collect_trian(matcontrib)...)
  end
  unique(trians)
end
filter_evaluation_function(u,args...) = u

function filter_evaluation_function(
  u::Gridap.ODEs.TransientFETools.TransientMultiFieldCellField,
  col::Int)

  u_col = Any[]
  for nf = eachindex(u.transient_single_fields)
    nf == col ? push!(u_col,u[col]) : push!(u_col,nothing)
  end
  u_col
end
