"""
A parametric version of the `Gridap` `TransientFEOperator`
"""
abstract type ParamTransientFEOperator{C<:OperatorType} <: GridapType end

"""
Returns a `ODEOperator` wrapper of the `ParamFEOperator` that can be
straightforwardly used with the `ODETools` module.
"""
function get_algebraic_operator(feop::ParamTransientFEOperator{C}) where C
  ParamODEOpFromFEOp{C}(feop)
end

function allocate_cache(::ParamTransientFEOperator)
  nothing
end

function update_cache!(
  ::Nothing,
  ::ParamTransientFEOperator,
  ::AbstractArray,
  ::Real)
  nothing
end

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
  assem = SparseMatrixAssembler(trial,test)
  ParamTransientFEOperatorFromWeakForm{Affine}(
    res,(jac,jac_t),assem,pspace,(trial,∂ₚt(trial)),test,1)
end

function ParamTransientFEOperator(res::Function,jac::Function,jac_t::Function,
  pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  ParamTransientFEOperatorFromWeakForm{Nonlinear}(
    res,(jac,jac_t),assem,pspace,(trial,∂ₚt(trial)),test,1)
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

function FESpaces.SparseMatrixAssembler(
  trial::Union{ParamTransientTrialFESpace,ParamTransientMultiFieldTrialFESpace},
  test::FESpace)
  SparseMatrixAssembler(trial(nothing,nothing),test)
end

get_test(op::ParamTransientFEOperatorFromWeakForm) = op.test

get_trial(op::ParamTransientFEOperatorFromWeakForm) = op.trials[1]

get_order(op::ParamTransientFEOperatorFromWeakForm) = op.order

get_pspace(op::ParamTransientFEOperatorFromWeakForm) = op.pspace

realization(op::ParamTransientFEOperator,args...) = realization(op.pspace,args...)

for OP in (:ParamTransientAffineFEOperator,:ParamTransientFEOperator)
  @eval begin
    function filter_operator(
      op::ParamTransientFEOperatorFromWeakForm,
      filter::NTuple{2,Int})

      if isa(get_test(op),MultiFieldFESpace)
        row,col = filter
        res = op.res
        jac,jac_t = op.jacs
        pspace = op.pspace
        trials_col = map(x->getindex(x,col),op.trials)
        test_row = getindex(op.test,row)
        return $OP(res,jac,jac_t,pspace,trials_col,test_row)
      else
        return op
      end
    end
  end
end

function allocate_residual(
  op::ParamTransientFEOperatorFromWeakForm,
  uh::T,
  cache) where T

  μ,t = realization(op),0.
  V = get_test(op)
  v = get_fe_basis(V)
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  vecdata = collect_cell_vector(V,evaluate(op.res(μ,t,xh,v)))
  allocate_vector(op.assem,vecdata)
end

function residual!(
  b::PTArray,
  op::ParamTransientFEOperatorFromWeakForm,
  μ::AbstractArray,
  t::Real,
  xh::T,
  cache) where T

  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,evaluate(op.res(μ,t,xh,v)))
  assemble_vector!(b,op.assem,vecdata)
  b
end

function allocate_jacobian(
  op::ParamTransientFEOperatorFromWeakForm,
  uh::T,
  cache) where T

  _matdata_jacobians = fill_initial_jacobians(op,uh)
  matdata = _vcat_matdata(_matdata_jacobians)
  allocate_matrix(op.assem,matdata)
end

for f in (:allocate_residual,:allocate_jacobian)
  @eval begin
    function $f(
      op::ParamTransientFEOperatorFromWeakForm,
      uh::PTCellField,
      cache)

      n = length(uh)
      uh1 = testitem(uh)
      a = $f(op,uh1,cache)
      PTArray(a,n)
    end
  end
end

function jacobian!(
  A::PTArray,
  op::ParamTransientFEOperatorFromWeakForm,
  μ::AbstractArray,
  t::Real,
  uh::T,
  i::Integer,
  γᵢ::Real,
  cache) where T

  matdata = _matdata_jacobian(op,μ,t,uh,i,γᵢ)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function jacobians!(
  A::PTArray,
  op::ParamTransientFEOperatorFromWeakForm,
  μ::AbstractArray,
  t::Real,
  uh::T,
  γ::Tuple{Vararg{Real}},
  cache) where T

  _matdata_jacobians = fill_jacobians(op,μ,t,uh,γ)
  matdata = _vcat_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function fill_initial_jacobians(
  op::ParamTransientFEOperatorFromWeakForm,
  uh::T) where T

  μ,t = realization(op),0.
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  _matdata = ()
  for i in 1:get_order(op)+1
    _matdata = (_matdata...,_matdata_jacobian(op,μ,t,xh,i,0.0))
  end
  return _matdata
end

function fill_jacobians(
  op::ParamTransientFEOperatorFromWeakForm,
  μ::AbstractArray,
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
  μ::AbstractArray,
  t::Real,
  xh::T,
  i::Integer,
  γᵢ::Real) where T

  Uh = get_trial(op)(μ,t)
  V = get_test(op)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  collect_cell_matrix(Uh,V,γᵢ*evaluate(op.jacs[i](μ,t,xh,u,v)))
end
