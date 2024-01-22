"""
A parametric version of the `Gridap` `TransientFEOperator`
"""
abstract type TransientPFEOperator{T<:OperatorType} <: GridapType end

function FESpaces.get_algebraic_operator(feop::TransientPFEOperator{C}) where C
  ODEPOpFromFEOp{C}(feop)
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
  tpspace::TransientParametricSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function AffineTransientPFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientPFEOperatorFromWeakForm{Affine}(
    res,(jac,jac_t),assem,tpspace,(trial,∂t(trial)),test,1)
end

function TransientPFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  TransientPFEOperatorFromWeakForm{Nonlinear}(
    res,(jac,jac_t),assem,tpspace,(trial,∂t(trial)),test,1)
end

struct NonlinearTransientPFEOperator <: TransientPFEOperator{Nonlinear}
  res::Function
  jacs::Tuple{Vararg{Function}}
  nl::Tuple{Vararg{Function}}
  assem::Assembler
  tpspace::ParametricSpace
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
        return $OP(res,jac,jac_t,op.tpspace,trials_col,test_row)
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
    return TransientPFEOperator(res,jac,jac_t,(nl,dnl),op.tpspace,trials_col,test_row)
  else
    return op
  end
end

FESpaces.get_test(op::TransientPFEOperatorFromWeakForm) = op.test
FESpaces.get_trial(op::TransientPFEOperatorFromWeakForm) = op.trials[1]
ReferenceFEs.get_order(op::TransientPFEOperatorFromWeakForm) = op.order
realization(op::TransientPFEOperatorFromWeakForm;kwargs...) = realization(op.tpspace;kwargs...)

function FESpaces.SparseMatrixAssembler(
  trial::TransientTrialPFESpace,
  test::FESpace)
  SparseMatrixAssembler(trial(nothing),test)
end

function Algebra.allocate_residual(
  op::TransientPFEOperatorFromWeakForm,
  r::TransientPRealization,
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
  assem = get_passembler(op.assem,r)
  vecdata = collect_cell_vector(test,dc)
  allocate_vector(assem,vecdata)
end

function Algebra.allocate_jacobian(
  op::TransientPFEOperatorFromWeakForm,
  r::TransientPRealization,
  uh::CellField,
  cache)

  _matdata_jacobians = TransientFETools.fill_initial_jacobians(op,r,uh)
  matdata = TransientFETools._vcat_matdata(_matdata_jacobians)
  assem = get_passembler(op.assem,r)
  allocate_matrix(assem,matdata)
end

function Algebra.allocate_jacobian(
  op::TransientPFEOperatorFromWeakForm,
  r::TransientPRealization,
  uh::CellField,
  i::Integer)

  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  trial = evaluate(get_trial(op),nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = op.jacs[i](get_parameters(r),get_times(r),xh,u,v)
  matdata = collect_cell_matrix(trial,test,dc)
  assem = get_passembler(op.assem,r)
  allocate_matrix(assem,matdata)
end

function Algebra.residual!(
  b::AbstractVector,
  op::TransientPFEOperatorFromWeakForm,
  r::TransientPRealization,
  xh::T,
  cache) where T

  test = get_test(op)
  v = get_fe_basis(test)
  dc = op.res(get_parameters(r),get_times(r),xh,v)
  vecdata = collect_cell_vector(test,dc)
  assem = get_passembler(op.assem,r)
  assemble_vector_add!(b,assem,vecdata)
  b
end

function residual_for_trian!(
  b::AbstractVector,
  op::TransientPFEOperatorFromWeakForm,
  r::TransientPRealization,
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
  r::TransientPRealization,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache) where T

  matdata = _matdata_jacobian(op,r,xh,i,γᵢ)
  assem = get_passembler(op.assem,r)
  assemble_matrix_add!(A,assem,matdata)
  A
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::TransientPFEOperatorFromWeakForm,
  r::TransientPRealization,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache,
  args...) where T

  trial = evaluate(get_trial(op),nothing)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = γᵢ*op.jacs[i](get_parameters(r),get_times(r),xh,u,v,args...)
  assemble_separate_matrix_add!(A,op,dc)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::TransientPFEOperatorFromWeakForm,
  r::TransientPRealization,
  xh::T,
  γ::Tuple{Vararg{Real}},
  cache) where T

  _matdata_jacobians = TransientFETools.fill_jacobians(op,r,xh,γ)
  matdata = TransientFETools._vcat_matdata(_matdata_jacobians)
  assem = get_passembler(op.assem,r)
  assemble_matrix_add!(A,assem,matdata)
  A
end

function TransientFETools.fill_initial_jacobians(
  op::TransientPFEOperatorFromWeakForm,
  r::TransientPRealization,
  uh::T) where T

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
  r::TransientPRealization,
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
  r::TransientPRealization,
  xh::T,
  i::Integer,
  γᵢ::Real) where T

  trial = evaluate(get_trial(op),nothing)
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
  assem = get_passembler(op.assem,r)
  bvec = Vector{typeof(b)}(undef,num_domains(dc))
  for (n,t) in enumerate(trian)
    vecdata = collect_cell_vector(test,dc,t)
    fill!(b,zero(eltype(b)))
    assemble_vector_add!(b,assem,vecdata)
    bvec[n] = copy(b)
  end
  bvec,trian
end

function assemble_separate_matrix_add!(
  A::AbstractMatrix,
  op::TransientPFEOperatorFromWeakForm,
  dc::DomainContribution)

  test = get_test(op)
  trial = get_trial(op)(nothing)
  trian = get_domains(dc)
  assem = get_passembler(op.assem,r)
  Avec = Vector{typeof(A)}(undef,num_domains(dc))
  for (n,t) in enumerate(trian)
    matdata = collect_cell_matrix(trial,test,dc,t)
    fillstored!(A,zero(eltype(A)))
    assemble_matrix_add!(A,assem,matdata)
    Avec[n] = copy(A)
  end
  Avec,trian
end


function TransientFETools.test_transient_fe_operator(op::TransientPFEOperator,uh,μt)
  odeop = get_algebraic_operator(op)
  @test isa(odeop,ODEPOperator)
  cache = allocate_cache(op)
  V = get_test(op)
  @test isa(V,FESpace)
  U = get_trial(op)
  U0 = U(μt)
  @test isa(U0,TrialPFESpace)
  r = allocate_residual(op,μt,uh,cache)
  @test isa(r,PArray{<:AbstractVector})
  xh = TransientCellField(uh,(uh,))
  residual!(r,op,μt,xh,cache)
  @test isa(r,PArray{<:AbstractVector})
  J = allocate_jacobian(op,μt,uh,cache)
  @test isa(J,PArray{<:AbstractMatrix})
  jacobian!(J,op,μt,xh,1,1.0,cache)
  @test isa(J,PArray{<:AbstractMatrix})
  jacobian!(J,op,μt,xh,2,1.0,cache)
  @test isa(J,PArray{<:AbstractMatrix})
  jacobians!(J,op,μt,xh,(1.0,1.0),cache)
  @test isa(J,PArray{<:AbstractMatrix})
  cache = update_cache!(cache,op,μt)
  true
end
