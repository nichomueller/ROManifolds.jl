"""
A parametric version of the `Gridap` `TransientFEOperator`
"""
abstract type PTFEOperator{C<:OperatorType} <: GridapType end

"""
Returns a `ODEOperator` wrapper of the `PFEOperator` that can be
straightforwardly used with the `ODETools` module.
"""
function TransientFETools.get_algebraic_operator(feop::PTFEOperator{C}) where C
  PODEOpFromFEOp{C}(feop)
end

function allocate_cache(::PTFEOperator)
  nothing
end

function update_cache!(
  ::Nothing,
  ::PTFEOperator,
  ::Any,
  ::Any)
  nothing
end

"""
Transient FE operator that is defined by a transient Weak form
"""
struct PTFEOperatorFromWeakForm{C<:OperatorType} <: PTFEOperator{C}
  res::Function
  jacs::Tuple{Vararg{Function}}
  assem::Assembler
  pspace::PSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function PTAffineFEOperator(res::Function,jac::Function,jac_t::Function,
  pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  PTFEOperatorFromWeakForm{Affine}(
    res,(jac,jac_t),assem,pspace,(trial,∂ₚt(trial)),test,1)
end

function PTFEOperator(res::Function,jac::Function,jac_t::Function,
  pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  PTFEOperatorFromWeakForm{Nonlinear}(
    res,(jac,jac_t),assem,pspace,(trial,∂ₚt(trial)),test,1)
end

function PTFEOperator(res::Function,pspace,trial,test;order::Integer=1)
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
  PTFEOperator(res,jacs...,pspace,trial,test)
end

function FESpaces.SparseMatrixAssembler(
  trial::Union{PTTrialFESpace,PTMultiFieldTrialFESpace},
  test::FESpace)
  SparseMatrixAssembler(trial(nothing,nothing),test)
end

get_test(op::PTFEOperator) = op.test

get_trial(op::PTFEOperator) = op.trials[1]

get_order(op::PTFEOperator) = op.order

get_pspace(op::PTFEOperator) = op.pspace

realization(op::PTFEOperator,args...) = realization(op.pspace,args...)

get_measure(op::PTFEOperator,trian::Triangulation) = Measure(trian,2*get_order(op.test))

for OP in (:PTAffineFEOperator,:PTFEOperator)
  @eval begin
    function filter_operator(
      op::PTFEOperatorFromWeakForm,
      idx::NTuple{2,Int})

      if isa(get_test(op),MultiFieldFESpace)
        row,col = idx
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
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  uh::S,
  cache) where {T,S}

  V = get_test(op)
  v = get_fe_basis(V)
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  dc = integrate(op.res(μ,t,xh,v),DomainContribution())
  vecdata = collect_cell_vector(V,dc)
  allocate_vector(op.assem,vecdata)
end

function allocate_jacobian(
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  uh::S,
  cache) where {T,S}

  _matdata_jacobians = fill_initial_jacobians(op,μ,t,uh)
  matdata = _vcat_matdata(_matdata_jacobians)
  allocate_matrix(op.assem,matdata)
end

for f in (:allocate_residual,:allocate_jacobian)
  @eval begin
    function $f(
      op::PTFEOperatorFromWeakForm,
      μ::AbstractVector,
      t::T,
      uh::PTCellField,
      cache) where T

      n = length(uh)
      μ1 = isa(μ,Table) ? testitem(μ) : μ
      t1 = isa(t,AbstractVector) ? testitem(t) : t
      uh1 = testitem(uh)
      a = $f(op,μ1,t1,uh1,cache)
      array = Vector{typeof(a)}(undef,n)
      @inbounds for i = eachindex(array)
        array[i] = copy(a)
      end
      PTArray(array)
    end
  end
end

function residual!(
  b::PTArray,
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  xh::S,
  cache) where {T,S}

  V = get_test(op)
  v = get_fe_basis(V)
  dc = integrate(op.res(μ,t,xh,v))
  vecdata = collect_cell_vector(V,dc)
  assemble_vector_add!(b,op.assem,vecdata)
  b
end

function residual!(
  b::PTArray,
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  xh::S,
  cache,
  meas::Measure) where {T,S}

  V = get_test(op)
  v = get_fe_basis(V)
  dc = op.res(μ,t,xh,v)[meas]
  vecdata = collect_cell_vector(V,dc)
  assemble_vector_add!(b,op.assem,vecdata)
  b
end

function residual_for_trian!(
  b::PTArray,
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  xh::S,
  cache) where {T,S}

  V = get_test(op)
  v = get_fe_basis(V)
  dc = integrate(op.res(μ,t,xh,v))
  trian = get_domains(dc)
  bvec = Vector{typeof(b)}(undef,num_domains(dc))
  for (n,t) in enumerate(trian)
    vecdata = collect_cell_vector(V,dc,t)
    assemble_vector_add!(b,op.assem,vecdata)
    bvec[n] = copy(b)
  end
  bvec,trian
end

function jacobian!(
  A::PTArray,
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  uh::S,
  i::Integer,
  γᵢ::Real,
  cache) where {T,S}

  matdata = _matdata_jacobian(op,μ,t,uh,i,γᵢ)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function jacobian!(
  A::PTArray,
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  uh::S,
  i::Integer,
  γᵢ::Real,
  cache,
  meas::Measure) where {T,S}

  Uh = get_trial(op)(μ,t)
  V = get_test(op)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  dc = γᵢ*op.jacs[i](μ,t,uh,u,v)[meas]
  matdata = collect_cell_matrix(Uh,V,dc)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function jacobian_for_trian!(
  A::PTArray,
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  uh::S,
  i::Integer,
  γᵢ::Real,
  cache) where {T,S}

  Uh = get_trial(op)(μ,t)
  V = get_test(op)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  dc = γᵢ*integrate(op.jacs[i](μ,t,uh,u,v))
  trian = get_domains(dc)
  Avec = Vector{typeof(A)}(undef,num_domains(dc))
  for (n,t) in enumerate(trian)
    matdata = collect_cell_matrix(Uh,V,dc,t)
    assemble_matrix_add!(A,op.assem,matdata)
    Avec[n] = copy(A)
  end
  Avec,trian
end

function jacobians!(
  A::PTArray,
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  uh::S,
  γ::Tuple{Vararg{Real}},
  cache) where {T,S}

  _matdata_jacobians = fill_jacobians(op,μ,t,uh,γ)
  matdata = _vcat_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function fill_initial_jacobians(
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  uh::S) where {T,S}

  cont = DomainContribution()
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  _matdata = ()
  for i in 1:get_order(op)+1
    _matdata = (_matdata...,_matdata_jacobian(op,μ,t,xh,i,0.0,cont))
  end
  return _matdata
end

function fill_jacobians(
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  uh::S,
  γ::Tuple{Vararg{Real}}) where {T,S}

  _matdata = ()
  for i in 1:get_order(op)+1
    if (γ[i] > 0.0)
      _matdata = (_matdata...,_matdata_jacobian(op,μ,t,uh,i,γ[i]))
    end
  end
  return _matdata
end

function _matdata_jacobian(
  op::PTFEOperatorFromWeakForm,
  μ::AbstractVector,
  t::T,
  xh::S,
  i::Integer,
  γᵢ::Real,
  args...) where {T,S}

  Uh = get_trial(op)(μ,t)
  V = get_test(op)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  dc = γᵢ*integrate(op.jacs[i](μ,t,xh,u,v),args...)
  collect_cell_matrix(Uh,V,dc)
end
