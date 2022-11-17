abstract type ParamFEOperator{C<:FunctionalStyle} <: GridapType end

# Default API

"""
Returns a `ParamOperator` wrapper of the `ParamFEOperator`
"""
function Gridap.FESpaces.get_algebraic_operator(feop::ParamFEOperator{C}) where C
  ParamOpFromFEOp{C}(feop)
end

# Specializations

"""
Parametric FE operator that is defined by a parametric weak form
"""
struct ParamFEOperatorFromWeakForm{C<:FunctionalStyle} <: ParamFEOperator{C}
  res::Function
  jac::Function
  assem::Assembler
  pparam::ParamSpace
  trial::ParamTrialFESpace
  test::FESpace
end

function ParamAffineFEOperator(a::Function,b::Function,pparam,trial,test)
  res(μ,u,v) = a(μ,u,v) - b(μ,v)
  jac(μ,u,du,v) = a(μ,du,v)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOperatorFromWeakForm{Affine}(res,jac,assem,pparam,trial,test)
end

function ParamFEOperator(res::Function,jac::Function,pparam,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOperatorFromWeakForm{Nonlinear}(res,jac,assem,pparam,trial,test)
end

function Gridap.FESpaces.SparseMatrixAssembler(
  trial::Union{ParamTrialFESpace,ParamMultiFieldTrialFESpace},
  test::FESpace)
  SparseMatrixAssembler(evaluate(trial,nothing),test)
end

get_assembler(op::ParamFEOperatorFromWeakForm) = op.assem
Gridap.FESpaces.get_test(op::ParamFEOperatorFromWeakForm) = op.test
Gridap.FESpaces.get_trial(op::ParamFEOperatorFromWeakForm) = op.trial

function Gridap.FESpaces.allocate_residual(
  op::ParamFEOperatorFromWeakForm,
  uh::T,
  args...) where T

  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(realization(op.pparam),uh,v))
  allocate_vector(op.assem,vecdata)
end

function Gridap.FESpaces.residual!(
  b::AbstractVector,
  op::ParamFEOperatorFromWeakForm,
  μ::Vector{Float},
  xh::T,
  args...) where T

  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(μ,xh,v))
  assemble_vector!(b,op.assem_t,vecdata)
  b
end

function Gridap.FESpaces.allocate_jacobian(
  op::ParamFEOperatorFromWeakForm,
  uh::CellField,
  args...)

  _matdata_jacobians = fill_initial_jacobians(op,uh)
  matdata = _vcat_matdata(_matdata_jacobians)
  allocate_matrix(op.assem_t,matdata)
end

function Gridap.FESpaces.jacobian!(
  A::AbstractMatrix,
  op::ParamFEOperatorFromWeakForm,
  μ::Vector{Float},
  xh::T,
  args...) where T

  matdata = _matdata_jacobian(op,μ,xh)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function _matdata_jacobian(
  op::ParamFEOperatorFromWeakForm,
  μ::Vector{Float},
  xh::T) where T

  Uh = evaluate(get_trial(op),nothing)
  V = get_test(op)
  du = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  collect_cell_matrix(Uh,V,op.jac(μ,xh,du,v))
end

"""
A parametric version of the `Gridap` `TransientFEOperator` that depends on a parameter μ
"""
abstract type
    ParamTransientFEOperator{C<:FunctionalStyle} <: TransientFEOperator{C} end

# Default API

"""
Returns a `ODEOperator` wrapper of the `ParamFEOperator` that can be
straightforwardly used with the `ODETools` module.
"""
function Gridap.ODEs.TransientFETools.get_algebraic_operator(
  feop::ParamTransientFEOperator{C}) where C

  ParamODEOpFromFEOp{C}(feop)
end

# Specializations

"""
Transient FE operator that is defined by a transient Weak form
"""
struct ParamTransientFEOperatorFromWeakForm{C} <: ParamTransientFEOperator{C}
  res::Function
  jacs::Tuple{Vararg{Function}}
  assem_t::Assembler
  pparam::ParamSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end

function ParamTransientAffineFEOperator(m::Function,a::Function,b::Function,
  pparam,trial,test)
  res(μ,t,u,v) = m(μ,t,∂t(u),v) + a(μ,t,u,v) - b(μ,t,v)
  jac(μ,t,u,du,v) = a(μ,t,du,v)
  jac_t(μ,t,u,dut,v) = m(μ,t,dut,v)
  assem_t = SparseMatrixAssembler(trial,test)
  ParamTransientFEOperatorFromWeakForm{Affine}(
    res,(jac,jac_t),assem_t,pparam,(trial,∂t(trial)),test,1)
end

function ParamTransientFEOperator(res::Function,jac::Function,jac_t::Function,
  pparam,trial,test)
  assem_t = SparseMatrixAssembler(trial,test)
  ParamTransientFEOperatorFromWeakForm{Nonlinear}(
    res,(jac,jac_t),assem_t,pparam,(trial,∂t(trial)),test,1)
end

function ParamTransientFEOperator(res::Function,pparam,trial,test;order::Integer=1)
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
  ParamTransientFEOperator(res,jacs...,pparam,trial,test)
end

function Gridap.ODEs.TransientFETools.SparseMatrixAssembler(
  trial::ParamTransientTrialFESpace,
  test::FESpace)
  SparseMatrixAssembler(evaluate(trial,nothing),test)
end

get_assembler(op::ParamTransientFEOperatorFromWeakForm) = op.assem_t
Gridap.ODEs.TransientFETools.get_test(op::ParamTransientFEOperatorFromWeakForm) = op.test
Gridap.ODEs.TransientFETools.get_trial(op::ParamTransientFEOperatorFromWeakForm) = op.trials[1]
Gridap.ODEs.TransientFETools.get_order(op::ParamTransientFEOperatorFromWeakForm) = op.order

function Gridap.ODEs.TransientFETools.allocate_residual(
  op::ParamTransientFEOperatorFromWeakForm,
  uh::T,
  cache) where T
  V = get_test(op)
  v = get_fe_basis(V)
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  vecdata = collect_cell_vector(V,op.res(0.0,xh,v))
  allocate_vector(op.assem_t,vecdata)
end

function Gridap.ODEs.TransientFETools.residual!(
  b::AbstractVector,
  op::ParamTransientFEOperatorFromWeakForm,
  μ::Vector{Float},
  t::Real,
  xh::T,
  cache) where T
  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(μ,t,xh,v))
  assemble_vector!(b,op.assem_t,vecdata)
  b
end

function Gridap.ODEs.TransientFETools.allocate_jacobian(
  op::ParamTransientFEOperatorFromWeakForm,
  uh::CellField,
  cache)
  _matdata_jacobians = fill_initial_jacobians(op,uh)
  matdata = _vcat_matdata(_matdata_jacobians)
  allocate_matrix(op.assem_t,matdata)
end

function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::ParamTransientFEOperatorFromWeakForm,
  μ::Vector{Float},
  t::Real,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache) where T
  matdata = _matdata_jacobian(op,μ,t,xh,i,γᵢ)
  assemble_matrix_add!(A,op.assem_t,matdata)
  A
end

function Gridap.ODEs.TransientFETools.jacobians!(
  A::AbstractMatrix,
  op::ParamTransientFEOperatorFromWeakForm,
  μ::Vector{Float},
  t::Real,
  xh::TransientCellField,
  γ::Tuple{Vararg{Real}},
  cache)
  _matdata_jacobians = fill_jacobians(op,μ,t,xh,γ)
  matdata = _vcat_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem_t,matdata)
  A
end

function Gridap.ODEs.TransientFETools.fill_initial_jacobians(
  op::ParamTransientFEOperatorFromWeakForm,uh)

  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  _matdata = ()
  for i in 1:get_order(op)+1
    _matdata = (_matdata...,_matdata_jacobian(op,realization(op.pparam),0.0,xh,i,0.0))
  end
  return _matdata
end

function Gridap.ODEs.TransientFETools.fill_jacobians(
  op::ParamTransientFEOperatorFromWeakForm,
  μ::Vector{Float},
  t::Real,
  xh::T,
  γ::Tuple{Vararg{Real}}) where T
  _matdata = ()
  for i in 1:get_order(op)+1
    if (γ[i] > 0.0)
      _matdata = (_matdata...,_matdata_jacobian(op,μ,t,xh,i,γ[i]))
    end
  end
  return _matdata
end

function _vcat_matdata(_matdata)
  term_to_cellmat_j = ()
  term_to_cellidsrows_j = ()
  term_to_cellidscols_j = ()
  for j in eachindex(_matdata)
    term_to_cellmat_j = (term_to_cellmat_j...,_matdata[j][1])
    term_to_cellidsrows_j = (term_to_cellidsrows_j...,_matdata[j][2])
    term_to_cellidscols_j = (term_to_cellidscols_j...,_matdata[j][3])
  end

  term_to_cellmat = vcat(term_to_cellmat_j...)
  term_to_cellidsrows = vcat(term_to_cellidsrows_j...)
  term_to_cellidscols = vcat(term_to_cellidscols_j...)

  (term_to_cellmat,term_to_cellidsrows, term_to_cellidscols)
end

function _matdata_jacobian(
  op::ParamTransientFEOperatorFromWeakForm,
  μ::Vector{Float},
  t::Real,
  xh::T,
  i::Integer,
  γᵢ::Real) where T
  Uh = evaluate(get_trial(op),nothing)
  V = get_test(op)
  du = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  collect_cell_matrix(Uh,V,γᵢ*op.jacs[i](μ,t,xh,du,v))
end
