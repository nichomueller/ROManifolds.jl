"""
A parametric version of the `Gridap` `TransientFEOperator`
"""
abstract type ParamTransientFEOperator{C<:OperatorType} <: GridapType end

"""
Transient FE operator that is defined by a transient Weak form
"""
struct ParamTransientFEOperatorFromWeakForm{C<:OperatorType} <: ParamTransientFEOperator{C}
  res::Function
  jacs::Tuple{Vararg{Function}}
  assem::Assembler
  pspace::ParamSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function ParamTransientAffineFEOperator(res::Function,jac::Function,jac_t::Function,
  pspace,trial,test)
  # res(μ,t,u,v) = m(μ,t,∂t(u),v) + a(μ,t,u,v) - b(μ,t,v)
  # jac(μ,t,u,du,v) = a(μ,t,du,v)
  # jac_t(μ,t,u,dut,v) = m(μ,t,dut,v)
  assem = SparseMatrixAssembler(trial,test)
  ParamTransientFEOperatorFromWeakForm{Affine}(
    res,(jac,jac_t),assem,pspace,(trial,∂t(trial)),test,1)
end

function ParamTransientFEOperator(res::Function,jac::Function,jac_t::Function,
  pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  ParamTransientFEOperatorFromWeakForm{Nonlinear}(
    res,(jac,jac_t),assem,pspace,(trial,∂t(trial)),test,1)
end

function ParamTransientFEOperator(res::Function,pspace,trial,test;order::Integer=1)
  function jac_0(μ,t,x,dx0,dv)
    function res_0(y)
      x0 = TransientCellField(y,x.derivatives)
      res(μ,t,x0,dv)
    end
    jacobian(res_0,x.cellfield)
  end
  jacs = (jac_0,)
  for i in 1:order
    function jac_i(μ,t,x,dxi,dv)
      function res_i(y)
        derivatives = (x.derivatives[1:i-1]...,y,x.derivatives[i+1:end]...)
        xi = TransientCellField(x.cellfield,derivatives)
        res(μ,t,xi,dv)
      end
      jacobian(res_i,x.derivatives[i])
    end
    jacs = (jacs...,jac_i)
  end
  ParamTransientFEOperator(res,jacs...,pspace,trial,test)
end

function Gridap.FESpaces.SparseMatrixAssembler(
  trial::Union{ParamTransientTrialFESpace,ParamTransientMultiFieldTrialFESpace},
  test::FESpace)
  SparseMatrixAssembler(trial(nothing,nothing),test)
end

get_test(op::ParamTransientFEOperatorFromWeakForm) = op.test

get_trial(op::ParamTransientFEOperatorFromWeakForm) = op.trials[1]

get_order(op::ParamTransientFEOperatorFromWeakForm) = op.order

get_pspace(op::ParamTransientFEOperatorFromWeakForm) = op.pspace

realization(op::ParamTransientFEOperator,args...) = realization(op.pspace,args...)

function allocate_cache(op::ParamTransientFEOperator)
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
  op::ParamTransientFEOperator,
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

function allocate_evaluation_function(op::ParamTransientFEOperator)
  μ,t = realization(op),1.
  trial = get_trial(op)(μ,t)
  xh = EvaluationFunction(trial,fill(0.,op.test.nfree))
  dxh = ()
  for _ in 1:get_order(op)
    dxh = (dxh...,xh)
  end
  TransientCellField(xh,dxh)
end

function evaluation_function(
  op::ParamTransientFEOperator,
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

for T in (:MultiFieldCellField,:TransientMultiFieldCellField)
  @eval begin
    function filter_evaluation_function(u::$T,col::Int)
      u_col = Any[]
      for nf = eachindex(u.transient_single_fields)
        nf == col ? push!(u_col,u[col]) : push!(u_col,nothing)
      end
      u_col
    end
  end
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

function _collect_trian_res(op::ParamTransientFEOperator)
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

function _collect_trian_jac(op::ParamTransientFEOperator)
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
