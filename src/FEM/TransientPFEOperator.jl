"""
A parametric version of the `Gridap` `TransientFEOperator`
"""
abstract type TransientPFEOperator{T<:OperatorType} <: GridapType end

function FESpaces.get_algebraic_operator(feop::TransientPFEOperator{C}) where C
  ODEOpFromFEOp{C}(feop)
end

function TransientFETools.allocate_cache(op::TransientPFEOperator)
  nothing
end

function TransientFETools.update_cache!(cache::Nothing,op::TransientPFEOperator,r)
  nothing
end

"""
Transient FE operator that is defined by a transient Weak form
"""
struct TransientPFEOperatorFromWeakForm{T<:OperatorType} <: TransientPFEOperator{T}
  res::Function
  jacs::Tuple{Vararg{Function}}
  assem::Assembler
  ptspace::TransientParametricSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function AffineTransientPFEOperator(
  res::Function,jac::Function,jac_t::Function,ptspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientPFEOperatorFromWeakForm{Affine}(
    res,(jac,jac_t),assem,ptspace,(trial,∂ₚt(trial)),test,1)
end

function TransientPFEOperator(
  res::Function,jac::Function,jac_t::Function,ptspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientPFEOperatorFromWeakForm{Nonlinear}(
    res,(jac,jac_t),assem,ptspace,(trial,∂ₚt(trial)),test,1)
end

struct NonlinearTransientPFEOperator <: TransientPFEOperator{Nonlinear}
  res::Function
  jacs::Tuple{Vararg{Function}}
  nl::Tuple{Vararg{Function}}
  assem::Assembler
  ptspace::ParametricSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function single_field(op::TransientPFEOperatorFromWeakForm,q,idx::Int)
  vq = Vector{Any}(undef,num_free_dofs(get_test(op)))
  fill!(vq,nothing)
  vq[idx] = q
  vq
end

function single_field(::TransientPFEOperatorFromWeakForm,q,::Colon)
  q
end

for (AFF,OP) in zip((:Affine,:Nonlinear),(:AffineTransientPFEOperator,:TransientPFEOperator))
  @eval begin
    function Base.getindex(op::TransientPFEOperatorFromWeakForm{$AFF},row,col)
      if isa(get_test(op),MultiFieldFESpace)
        trials_col = get_trial(op)[col]
        test_row = op.test[row]
        sf(q,idx) = single_field(op,q,idx)
        res(μ,t,u,dv) = op.res(μ,t,sf(u,col),sf(dv,row))
        jac(μ,t,u,du,dv) = op.jacs[1](μ,t,sf(u,col),sf(du,col),sf(dv,row))
        jac_t(μ,t,u,dut,dv) = op.jacs[2](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
        return $OP(res,jac,jac_t,op.ptspace,trials_col,test_row)
      else
        return op
      end
    end
  end
end

function Base.getindex(op::NonlinearTransientPFEOperator,row,col)
  if isa(get_test(op),MultiFieldFESpace)
    trials_col = get_trial(op)[col]
    test_row = op.test[row]
    sf(q,idx) = single_field(op,q,idx)
    res(μ,t,u,dv) = op.res(μ,t,sf(u,col),sf(dv,row))
    jac(μ,t,u,du,dv) = op.jacs[1](μ,t,sf(u,col),sf(du,col),sf(dv,row))
    jac_t(μ,t,u,dut,dv) = op.jacs[2](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
    nl(μ,t,u,dut,dv) = op.nl[1](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
    dnl(μ,t,u,dut,dv) = op.nl[2](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
    return TransientPFEOperator(res,jac,jac_t,(nl,dnl),op.ptspace,trials_col,test_row)
  else
    return op
  end
end

FESpaces.get_test(op::TransientPFEOperatorFromWeakForm) = op.test
FESpaces.get_trial(op::TransientPFEOperatorFromWeakForm) = op.trials[1]
ReferenceFEs.get_order(op::TransientPFEOperatorFromWeakForm) = op.order
realization(op::TransientPFEOperatorFromWeakForm,args...) = realization(op.ptspace,args...)

function Algebra.allocate_residual(
  op::TransientPFEOperatorFromWeakForm,
  r::Realization,
  uh::T,
  cache) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  dc = op.res(get_parameters(r),get_times(r),xh,v)
  vecdata = collect_cell_vector(test,dc)
  allocate_vector(op.assem,vecdata)
end

function Algebra.allocate_jacobian(
  op::TransientPFEOperatorFromWeakForm,
  r::Realization,
  uh::CellField,
  cache)

  _matdata_jacobians = fill_initial_jacobians(op,r,uh,cache)
  matdata = _vcat_matdata(_matdata_jacobians)
  allocate_matrix(op.assem,matdata)
end

function Algebra.allocate_jacobian(
  op::TransientPFEOperator,
  r::Realization,
  uh::CellField,
  i::Integer)

  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  trial = evaluate(get_trial(op),nothing,nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = op.jacs[i](get_parameters(r),get_times(r),xh,u,v)
  matdata = collect_cell_matrix(trial,test,dc)
  allocate_matrix(op.assem,matdata)
end

function Algebra.residual!(
  b::AbstractVector,
  op::TransientPFEOperatorFromWeakForm,
  r::Realization,
  xh::T,
  cache) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dc = op.res(get_parameters(r),get_times(r),xh,v)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,op.assem,vecdata)
  b
end

function residual_for_trian!(
  b::AbstractVector,
  op::TransientPFEOperatorFromWeakForm,
  r::Realization,
  xh::T,
  cache,
  args...) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dc = op.res(get_parameters(r),get_times(r),xh,v,args...)
  assemble_separate_vector_add!(b,op,dc)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::TransientPFEOperatorFromWeakForm,
  r::Realization,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache) where T

  matdata = _matdata_jacobian(op,r,xh,i,γᵢ)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::TransientPFEOperatorFromWeakForm,
  r::Realization,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache,
  args...) where T

  trial = evaluate(get_trial(op),nothing,nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = γᵢ*op.jacs[i](get_parameters(r),get_times(r),xh,u,v,args...)
  assemble_separate_matrix_add!(A,op,dc)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::TransientPFEOperatorFromWeakForm,
  r::Realization,
  xh::T,
  γ::Tuple{Vararg{Real}},
  cache) where T

  _matdata_jacobians = fill_jacobians(op,r,xh,γ)
  matdata = _vcat_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function TransientFETools.fill_initial_jacobians(
  op::TransientFEOperatorsFromWeakForm,
  r::Realization,
  xh::T) where T

  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  _matdata = ()
  for i in 1:get_order(op)+1
    _data = _matdata_jacobian(op,r,xh,i,0.0)
    if !isnothing(_data)
      _matdata = (_matdata...,_data)
    end
  end
  return _matdata
end

function TransientFETools.fill_jacobians(
  op::TransientPFEOperatorFromWeakForm,
  r::Realization,
  xh::T,
  γ::Tuple{Vararg{Real}}) where T

  _matdata = ()
  for i in 1:get_order(op)+1
    if (γ[i] > 0.0)
      _data = _matdata_jacobian(op,r,xh,i,γ[i])
      if !isnothing(_data)
        _matdata = (_matdata...,_data)
      end
    end
  end
  return _matdata
end

function TransientFETools._matdata_jacobian(
  op::TransientPFEOperatorFromWeakForm,
  r::Realization,
  xh::T,
  i::Integer,
  γᵢ::Real) where T

  trial = evaluate(get_trial(op),nothing,nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = γᵢ*op.jacs[i](get_parameters(r),get_times(r),xh,u,v)
  collect_cell_matrix(trial,test,dc)
end

function assemble_separate_vector_add!(
  b::AbstractVector,
  op::TransientPFEOperatorFromWeakForm,
  dc::DomainContribution)

  test = get_test(op)
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

function assemble_separate_matrix_add!(
  A::AbstractMatrix,
  op::TransientPFEOperatorFromWeakForm,
  dc::DomainContribution)

  test = get_test(op)
  trial = get_trial(op)(nothing,nothing)
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
