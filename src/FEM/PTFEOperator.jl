"""
A parametric version of the `Gridap` `TransientFEOperator`
"""
abstract type PTFEOperator{T<:OperatorType} <: GridapType end

"""
Returns a `ODEOperator` wrapper of the `PFEOperator` that can be
straightforwardly used with the `ODETools` module.
"""
function TransientFETools.get_algebraic_operator(feop::PTFEOperator{T}) where T
  PODEOpFromFEOp{T}(feop)
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

get_test(op::PTFEOperator) = op.test
get_trial(op::PTFEOperator) = op.trials[1]
get_order(op::PTFEOperator) = op.order
get_pspace(op::PTFEOperator) = op.pspace
realization(op::PTFEOperator,args...) = realization(op.pspace,args...)
get_measure(op::PTFEOperator,trian::Triangulation) = Measure(trian,2*get_order(op.test))

"""
Transient FE operator that is defined by a transient Weak form
"""
struct PTFEOperatorFromWeakForm <: PTFEOperator{Affine}
  res::Function
  jacs::Tuple{Vararg{Function}}
  assem::Assembler
  pspace::PSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function AffinePTFEOperator(
  res::Function,jac::Function,jac_t::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  PTFEOperatorFromWeakForm(res,(jac,jac_t),assem,pspace,(trial,∂ₚt(trial)),test,1)
end

function get_residual(op::PTFEOperatorFromWeakForm)
  return op.res
end

function get_jacobian(op::PTFEOperatorFromWeakForm)
  return op.jacs
end

struct NonlinearPTFEOperator <: PTFEOperator{Nonlinear}
  res::Function
  jacs::Tuple{Vararg{Function}}
  nl::Tuple{Vararg{Function}}
  assem::Assembler
  pspace::PSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function NonlinearPTFEOperator(
  res::Function,jac::Function,jac_t::Function,nl::Tuple{Vararg{Function}},pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  NonlinearPTFEOperator(res,(jac,jac_t),nl,assem,pspace,(trial,∂ₚt(trial)),test,1)
end

function get_residual(op::NonlinearPTFEOperator)
  res(μ,t,u,v) = op.res(μ,t,u,v) + op.nl[1](μ,t,u,u,v)
  return res
end

function get_jacobian(op::NonlinearPTFEOperator)
  jac(μ,t,u,du,v) = op.jacs[1](μ,t,u,du,v) + op.nl[2](μ,t,u,du,v)
  jac_t(μ,t,u,du,v) = op.jacs[2](μ,t,u,du,v)
  return jac,jac_t
end

function get_linear_operator(op::NonlinearPTFEOperator)
  res(μ,t,u,v) = op.res(μ,t,u,v)
  jac(μ,t,u,du,v) = op.jacs[1](μ,t,u,du,v) + op.nl[1](μ,t,u,du,v)
  jac_t(μ,t,u,du,v) = op.jacs[2](μ,t,u,du,v)
  return PTFEOperatorFromWeakForm(res,(jac,jac_t),op.assem,op.pspace,op.trials,op.test,op.order)
end

function get_nonlinear_operator(op::NonlinearPTFEOperator)
  res(μ,t,u,v) = op.nl[1](μ,t,u,u,v)
  jac(μ,t,u,du,v) = op.nl[2](μ,t,u,du,v) - op.nl[1](μ,t,u,du,v)
  jac_t(μ,t,u,du,v) = nothing
  return PTFEOperatorFromWeakForm(res,(jac,jac_t),op.assem,op.pspace,op.trials,op.test,op.order)
end

function single_field(op::PTFEOperator,q,idx::Int)
  vq = Any[]
  for i in eachindex(get_test(op).spaces)
    if i == idx
      push!(vq,q)
    else
      push!(vq,nothing)
    end
  end
  vq
end

function single_field(::PTFEOperator,q,::Colon)
  q
end

function Base.getindex(op::PTFEOperatorFromWeakForm,row,col)
  if isa(get_test(op),MultiFieldFESpace)
    trials_col = get_trial(op)[col]
    test_row = op.test[row]
    sf(q,idx) = single_field(op,q,idx)
    res(μ,t,u,dv) = op.res(μ,t,sf(u,col),sf(dv,row))
    jac(μ,t,u,du,dv) = op.jacs[1](μ,t,sf(u,col),sf(du,col),sf(dv,row))
    jac_t(μ,t,u,dut,dv) = op.jacs[2](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
    return AffinePTFEOperator(res,jac,jac_t,op.pspace,trials_col,test_row)
  else
    return op
  end
end

function Base.getindex(op::NonlinearPTFEOperator,row,col)
  if isa(get_test(op),MultiFieldFESpace)
    trials_col = get_trial(op)[col]
    test_row = op.test[row]
    sf(q,idx) = single_field(op,q,idx)
    res(μ,t,u,dv) = op.res(μ,t,sf(u,col),sf(dv,row))
    jac(μ,t,u,du,dv) = op.jacs[1](μ,t,sf(u,col),sf(du,col),sf(dv,row))
    jac_t(μ,t,u,dut,dv) = op.jacs[2](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
    nl(μ,t,u,dut,dv) = op.nl[1](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
    dnl(μ,t,u,dut,dv) = op.nl[2](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
    return NonlinearPTFEOperator(res,jac,jac_t,(nl,dnl),op.pspace,trials_col,test_row)
  else
    return op
  end
end

function allocate_residual(
  op::PTFEOperator,
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
  res = get_residual(op)
  dc = integrate(res(μ,t,xh,v))
  vecdata = collect_cell_vector(V,dc)
  allocate_vector(op.assem,vecdata)
end

function allocate_jacobian(
  op::PTFEOperator,
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
      op::PTFEOperator,
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
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  xh::S,
  cache) where {T,S}

  V = get_test(op)
  v = get_fe_basis(V)
  res = get_residual(op)
  dc = integrate(res(μ,t,xh,v))
  vecdata = collect_cell_vector(V,dc)
  assemble_vector_add!(b,op.assem,vecdata)
  b
end

function residual!(
  b::PTArray,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  xh::S,
  cache,
  meas::Measure) where {T,S}

  V = get_test(op)
  v = get_fe_basis(V)
  res = get_residual(op)
  dc = res(μ,t,xh,v)[meas]
  vecdata = collect_cell_vector(V,dc)
  assemble_vector_add!(b,op.assem,vecdata)
  b
end

function residual_for_trian!(
  b::PTArray,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  xh::S,
  cache) where {T,S}

  V = get_test(op)
  v = get_fe_basis(V)
  res = get_residual(op)
  dc = integrate(res(μ,t,xh,v))
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
  op::PTFEOperator,
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
  op::PTFEOperator,
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
  jac = get_jacobian(op)
  dc = γᵢ*jac[i](μ,t,uh,u,v)[meas]
  matdata = collect_cell_matrix(Uh,V,dc)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function jacobian_for_trian!(
  A::PTArray,
  op::PTFEOperator,
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
  jac = get_jacobian(op)
  dc = γᵢ*integrate(jac[i](μ,t,uh,u,v))
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
  op::PTFEOperator,
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
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::S) where {T,S}

  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  _matdata = ()
  for i in 1:get_order(op)+1
    _data = _matdata_jacobian(op,μ,t,xh,i,0.0)
    if !isnothing(_data)
      _matdata = (_matdata...,_matdata_jacobian(op,μ,t,xh,i,0.0))
    end
  end
  return _matdata
end

function fill_jacobians(
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::S,
  γ::Tuple{Vararg{Real}}) where {T,S}

  _matdata = ()
  for i in 1:get_order(op)+1
    if (γ[i] > 0.0)
      _data = _matdata_jacobian(op,μ,t,uh,i,γ[i])
      if !isnothing(_data)
        _matdata = (_matdata...,_data)
      end
    end
  end
  return _matdata
end

function _matdata_jacobian(
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  xh::S,
  i::Integer,
  γᵢ::Real) where {T,S}

  Uh = get_trial(op)(μ,t)
  V = get_test(op)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  jac = get_jacobian(op)
  dc = γᵢ*integrate(jac[i](μ,t,xh,u,v))
  collect_cell_matrix(Uh,V,dc)
end
