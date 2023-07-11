abstract type ParamSolution <: GridapType end

mutable struct GenericParamSolution <: ParamSolution
  uh::AbstractArray
  μ::AbstractVector
end

function solve!(
  op::ParamFEOperator,
  solver::FESolver,
  xh::AbstractVector,
  μ::AbstractVector)

  A,b = _allocate_matrix_and_vector(op,xh)
  residual!(b,op,xh,μ)
  jacobian!(A,op,xh,μ)
  dx = similar(b)
  ss = symbolic_setup(solver,A)
  ns = numerical_setup(ss,A)
  nls = NonlinearFESolver(solver)
  Gridap.Algebra._solve_nr!(xh,A,b,dx,ns,nls,op)
  xh
end

function solve!(
  op::ParamFEOperator{Affine},
  solver::FESolver,
  xh::AbstractVector,
  μ::AbstractVector)

  A,b = _allocate_matrix_and_vector(op,xh)
  A = _matrix!(A,op,xh,μ)
  b = _vector!(b,op,xh,μ)
  afop = AffineOperator(A,b)
  cache = nothing
  newmatrix = true
  solve!(xh,solver.ls,afop,cache,newmatrix)
  xh
end

function solve(
  op::ParamFEOperator,
  solver::FESolver,
  μ::AbstractVector)

  xh = zero(op.test.nfree)
  solk = solve!(op,solver,xh,μ)
  GenericParamSolution(solk,μ)
end
