"""
A parametric version of the `Gridap` `TransientFEOperator`
"""
abstract type PTFEOperator{T<:OperatorType} <: GridapType end

function TransientFETools.allocate_cache(
  op::PTFEOperator,
  μ::AbstractVector,
  t::T) where T

  Ut = get_trial(op)
  U = allocate_trial_space(Ut,μ,t)
  Uts = (Ut,)
  Us = (U,)
  for i in 1:get_order(op)
    Uts = (Uts...,∂ₚt(Uts[i]))
    Us = (Us...,allocate_trial_space(Uts[i+1],μ,t))
  end
  fecache = nothing
  Us,Uts,fecache
end

function TransientFETools.update_cache!(
  ode_cache,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T) where T

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],μ,t))
  end
  fecache = nothing
  Us,Uts,fecache
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

function Algebra.allocate_residual(
  op::PTFEOperator,
  μ::P,
  t::T,
  xh::CellField) where {P,T}

  test = get_test(op)
  v = get_fe_basis(test)
  dc = integrate(op.res(μ,t,xh,v))
  vecdata = collect_cell_vector(test,dc)
  assem = op.assem
  for trian in get_domains(dc)
    if typeof(dc[trian]) <: AbstractArray{<:PTArray}
      assem = PTSparseMatrixAssembler(assem,μ,t)
      break
    end
  end
  allocate_vector(assem,vecdata)
end

function Algebra.allocate_jacobian(
  op::PTFEOperator,
  μ::P,
  t::T,
  xh::CellField,
  i::Integer) where {P,T}

  trial = get_trial(op)(μ,t)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = integrate(op.jacs[i](μ,t,xh,u,v))
  matdata = collect_cell_matrix(trial,test,dc)
  assem = op.assem
  for trian in get_domains(dc)
    if typeof(dc[trian]) <: AbstractArray{<:PTArray}
      assem = PTSparseMatrixAssembler(assem,μ,t)
      break
    end
  end
  allocate_matrix(assem,matdata)
end

function Algebra.residual!(
  b::AbstractVector,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  xh::CellField) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dc = integrate(op.res(μ,t,xh,v))
  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,op.assem,vecdata)
  b
end

function residual_for_trian!(
  b::AbstractVector,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  xh::CellField,
  args...) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dc = integrate(op.res(μ,t,xh,v),args...)
  trian = get_domains(dc)
  bvec = Vector{typeof(b)}(undef,num_domains(dc))
  for (n,t) in enumerate(trian)
    vecdata = collect_cell_vector(test,dc,t)
    fill!(b,zero(eltype(b)))
    assemble_vector_add!(b,op.assem,vecdata)
    bvec[n] = copy(b)
  end
  bvec,trian
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::CellField,
  i::Integer,
  γᵢ::Real) where T

  matdata = _matdata_jacobian(op,μ,t,uh,i,γᵢ)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::CellField,
  i::Integer,
  γᵢ::Real,
  args...) where T

  trial = get_trial(op)(μ,t)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = γᵢ*integrate(op.jacs[i](μ,t,uh,u,v),args...)
  trian = get_domains(dc)
  Avec = Vector{typeof(A)}(undef,num_domains(dc))
  for (n,t) in enumerate(trian)
    matdata = collect_cell_matrix(trial,test,dc,t)
    fillstored!(A,zero(eltype(A)))
    assemble_matrix_add!(A,op.assem,matdata)
    Avec[n] = copy(A)
  end
  Avec,trian
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::CellField,
  γ::Tuple{Vararg{Real}}) where T

  _matdata_jacobians = fill_jacobians(op,μ,t,uh,γ)
  matdata = _vcat_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function TransientFETools.fill_jacobians(
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::CellField,
  γ::Tuple{Vararg{Real}}) where T

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
  xh::CellField,
  i::Integer,
  γᵢ::Real) where T

  trial = get_trial(op)(μ,t)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = γᵢ*integrate(op.jacs[i](μ,t,xh,u,v))
  collect_cell_matrix(trial,test,dc)
end
