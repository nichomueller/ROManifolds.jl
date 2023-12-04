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

function TransientFETools.allocate_cache(::PTFEOperator)
  nothing
end

function Gridap.ODEs.TransientFETools.update_cache!(
  ::Nothing,
  ::PTFEOperator,
  ::Any,
  ::Any)
  nothing
end

FESpaces.get_test(op::PTFEOperator) = op.test
FESpaces.get_trial(op::PTFEOperator) = op.trials[1]
ReferenceFEs.get_order(op::PTFEOperator) = op.order
realization(op::PTFEOperator,args...) = realization(op.pspace,args...)

"""
Transient FE operator that is defined by a transient Weak form
"""
struct PTFEOperatorFromWeakForm{T<:OperatorType} <: PTFEOperator{T}
  res::Function
  jacs::Tuple{Vararg{Function}}
  assem::Assembler
  pspace::PSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function AffinePTFEOperator(res::Function,jac::Function,jac_t::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  PTFEOperatorFromWeakForm{Affine}(res,(jac,jac_t),assem,pspace,(trial,∂ₚt(trial)),test,1)
end

function PTFEOperator(res::Function,jac::Function,jac_t::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  PTFEOperatorFromWeakForm{Nonlinear}(res,(jac,jac_t),assem,pspace,(trial,∂ₚt(trial)),test,1)
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

function PTFEOperator(
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

function linear_operator(op::NonlinearPTFEOperator)
  res(μ,t,u,v) = op.res(μ,t,u,v)
  jac(μ,t,u,du,v) = op.jacs[1](μ,t,u,du,v)
  jac_t(μ,t,u,du,v) = op.jacs[2](μ,t,u,du,v)
  return PTFEOperatorFromWeakForm{Affine}(res,(jac,jac_t),op.assem,op.pspace,op.trials,op.test,op.order)
end

function nonlinear_operator(op::NonlinearPTFEOperator)
  u0(μ,t) = zero(op.trials[1](μ,t))
  res(μ,t,u,v) = op.nl[1](μ,t,u,u0(μ,t),v)
  jac(μ,t,u,du,v) = op.nl[2](μ,t,u,du,v)
  jac_t(μ,t,u,du,v) = nothing
  return PTFEOperatorFromWeakForm{Nonlinear}(res,(jac,jac_t),op.assem,op.pspace,op.trials,op.test,op.order)
end

function auxiliary_operator(op::NonlinearPTFEOperator)
  res(μ,t,u,v) = nothing
  jac(μ,t,u,du,v) = op.nl[1](μ,t,u,du,v)
  jac_t(μ,t,u,du,v) = nothing
  return PTFEOperatorFromWeakForm{Nonlinear}(res,(jac,jac_t),op.assem,op.pspace,op.trials,op.test,op.order)
end

function single_field(op::PTFEOperator,q,idx::Int)
  vq = Vector{Any}(undef,num_free_dofs(get_test(op)))
  fill!(vq,nothing)
  vq[idx] = q
  vq
end

function single_field(::PTFEOperator,q,::Colon)
  q
end

for (AFF,OP) in zip((:Affine,:Nonlinear),(:AffinePTFEOperator,:PTFEOperator))
  @eval begin
    function Base.getindex(op::PTFEOperatorFromWeakForm{$AFF},row,col)
      if isa(get_test(op),MultiFieldFESpace)
        trials_col = get_trial(op)[col]
        test_row = op.test[row]
        sf(q,idx) = single_field(op,q,idx)
        res(μ,t,u,dv) = op.res(μ,t,sf(u,col),sf(dv,row))
        jac(μ,t,u,du,dv) = op.jacs[1](μ,t,sf(u,col),sf(du,col),sf(dv,row))
        jac_t(μ,t,u,dut,dv) = op.jacs[2](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
        return $OP(res,jac,jac_t,op.pspace,trials_col,test_row)
      else
        return op
      end
    end
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
    return PTFEOperator(res,jac,jac_t,(nl,dnl),op.pspace,trials_col,test_row)
  else
    return op
  end
end

function Gridap.Algebra.allocate_residual(
  op::PTFEOperator,
  μ::P,
  t::T,
  uh::PTCellField,
  cache) where {P,T}

  V = get_test(op)
  v = get_fe_basis(V)
  dxh = ()
  for _ in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  res = get_residual(op)
  dc = integrate(res(μ,t,xh,v))
  dc1 = testitem(dc)
  vecdata1 = collect_cell_vector(V,dc1)
  allocate_vector(dc,op.assem,vecdata1;N=length(uh))
end

function Gridap.Algebra.allocate_jacobian(
  op::PTFEOperator,
  μ::P,
  t::T,
  uh::PTCellField,
  i::Integer,
  cache) where {P,T}

  Uh = get_trial(op)(μ,t)
  V = get_test(op)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  dxh = ()
  for _ in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  jac = get_jacobian(op)
  dc = integrate(jac[i](μ,t,xh,u,v))
  dc1 = testitem(dc)
  matdata1 = collect_cell_matrix(Uh,V,dc1)
  allocate_matrix(dc,op.assem,matdata1;N=length(uh))
end

function Algebra.residual!(
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

function residual_for_trian!(
  b::PTArray,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  xh::S,
  cache,
  args...) where {T,S}

  V = get_test(op)
  v = get_fe_basis(V)
  res = get_residual(op)
  dc = integrate(res(μ,t,xh,v),args...)
  trian = get_domains(dc)
  bvec = Vector{typeof(b)}(undef,num_domains(dc))
  for (n,t) in enumerate(trian)
    vecdata = collect_cell_vector(V,dc,t)
    assemble_vector_add!(b,op.assem,vecdata)
    bvec[n] = copy(b)
  end
  bvec,trian
end

function Algebra.jacobian!(
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

function jacobian_for_trian!(
  A::PTArray,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::S,
  i::Integer,
  γᵢ::Real,
  cache,
  args...) where {T,S}

  Uh = get_trial(op)(μ,t)
  V = get_test(op)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  jac = get_jacobian(op)
  dc = γᵢ*integrate(jac[i](μ,t,uh,u,v),args...)
  trian = get_domains(dc)
  Avec = Vector{typeof(A)}(undef,num_domains(dc))
  for (n,t) in enumerate(trian)
    matdata = collect_cell_matrix(Uh,V,dc,t)
    assemble_matrix_add!(A,op.assem,matdata)
    Avec[n] = copy(A)
  end
  Avec,trian
end

function ODETools.jacobians!(
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

function TransientFETools.fill_jacobians(
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

function TransientFETools._matdata_jacobian(
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
